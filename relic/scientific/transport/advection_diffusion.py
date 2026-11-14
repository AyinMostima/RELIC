"""
Advection-Diffusion Transport Module for RELIC

This module implements combined advection-diffusion transport for scalar
quantities, providing unified time stepping with operator-owned stability
rejection and operator splitting methods. Uses unified operators from operators.py for
efficient and consistent transport calculations.

Mathematical Expression:
    Advection-Diffusion equation: partialc/partialt + nabla*(vc) = nabla*(Knablac) + S
    where:
    - c is concentration [kg m-3]
    - v is velocity field [m s-1]
    - K is diffusivity [m^2 s-1]
    - S is source/sink term [kg m-3 s-1]

    Combined stability constraint:
    - CFL: |v| * Deltat / Deltax <= 1.0 (advection)
    - Diffusion: K * Deltat / Deltax^2 <= 0.5 (diffusion)
    - Combined: Deltat <= min(Deltax/|v|, 0.5*Deltax^2/K)

References:
    - Hundsdorfer, W. & Verwer, J.G. (2003). Numerical Solution of Time-Dependent
      Advection-Diffusion-Reaction Equations. Springer. ISBN: 978-3-662-09017-6
    - Morton, K.W. & Mayers, D.F. (2005). Numerical Solution of Partial Differential
      Equations: An Introduction. Cambridge University Press. ISBN: 978-0-521-60793-3
    - LeVeque, R.J. (2007). Finite Difference Methods for Ordinary and Partial
      Differential Equations. SIAM. ISBN: 978-0-898716-29-0
    - Strikwerda, J.C. (2004). Finite Difference Schemes and Partial Differential
      Equations (2nd ed.). SIAM. ISBN: 978-0-898715-67-7
    - Strang, G. (1968). On the construction and comparison of difference schemes.
      SIAM Journal on Numerical Analysis, 5(3), 506-517. DOI: 10.1137/0705041
"""

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, cast

import torch
from torch import Tensor

from ...core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from ...core.functor import TransportMechanismFunctor
from ...core.state import UpdateItem, UpdateMode

# Import unified transport operators from computation layer
from ...computation.operators import OperatorFactory
from ...computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from ...computation.operators.functional import pack_outward_boundary_flux_cochain
from ...computation.operators.iphs import PortSpecIPHS

# Type aliases
StateDict = Dict[str, Tensor]
EnvironmentDict = Dict[str, Tensor]
FluxDict = Dict[str, Any]
GridSpacing = Union[float, Tuple[float, ...]]
VelocityField = Union[Tensor, Tuple[Tensor, ...]]


class AdvectionDiffusion(TransportMechanismFunctor):
    """
    Combined advection-diffusion transport mechanism.

    Implements the full advection-diffusion equation using unified operators
    from operators.py. The canonical operator rejects unstable scheduler steps
    and retains every nonzero boundary exchange as an oriented face cochain.

    Mathematical Expression:
        Full equation: partialc/partialt + nabla*(vc) = nabla*(Knablac) + S

        where c is concentration, v is velocity, K is diffusivity, S is source

    Parameters:
        spatial_dims : int
            Number of spatial dimensions (1, 2, or 3)
        diffusivity : Union[float, Tensor]
            Diffusion coefficient [m^2 s-1]
        velocity_field : Optional[VelocityField]
            Velocity field(s) [m s-1]
        grid_spacing : GridSpacing
            Grid spacing(s) [m]
        splitting_method : str
            Time splitting method: "sequential", "symmetric", "combined"
        boundary_method : Optional[str]
            Explicit boundary configuration; ParameterManager supplies the default
        track_peclet : bool
            Whether to track and report Peclet numbers for diagnostics
        device : Optional[torch.device]
            Computation device
    """

    TRANSPORT_TYPE = "advection_diffusion"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"

    def __init__(
        self,
        spatial_dims: int = 1,
        diffusivity: Optional[Union[float, Tensor]] = None,
        velocity_field: Optional[VelocityField] = None,
        grid_spacing: Optional[GridSpacing] = None,
        splitting_method: Optional[str] = None,
        advection_scheme: str = "upwind",
        boundary_method: Optional[str] = None,
        non_negative: Optional[bool] = None,
        track_peclet: bool = False,
        device: Optional[torch.device] = None,
        name: str = "advection_diffusion",
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[FunctorMetadata] = None,
    ) -> None:
        """
        Initialize advection-diffusion transport mechanism.
        """
        # Initialize TransportMechanism parent class
        functor_name = name
        if metadata is None:
            metadata = FunctorMetadata(
                name=functor_name,
                version="1.0.0",
                description="Combined advection-diffusion transport with operator-owned stability guards.",
                category=ProcessCategory.TRANSPORT_MECHANISM,
                execution_phase=ExecutionPhase.MAIN_PROCESSES,
                mathematical_expression=(
                    "c_dot = -D_h(q_adv_h + q_diff_h) + G_h u_h; "
                    "q_adv_h = i_v c_h; q_diff_h = -star_K d_h c_h; "
                    "G_h u_h is source_term plus routed open flux"
                ),
                references=[
                    "Hundsdorfer, W. and Verwer, J.G. (2003). Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations.",
                    "Morton, K.W. and Mayers, D.F. (2005). Numerical Solution of Partial Differential Equations.",
                    "LeVeque, R.J. (2007). Finite Difference Methods for Ordinary and Partial Differential Equations.",
                    "Strang, G. (1968). On the construction and comparison of difference schemes."
                ],
                parameters={
                    "transport.advection.default_velocity_magnitude": "Structural Hodge audit speed [m s-1]",
                    "transport.advection_diffusion.splitting_method": "Canonical operator splitting scheme [dimensionless]",
                    "transport.diffusion.diffusivity": "Fickian diffusion coefficient [m2 s-1]",
                    "transport.advection_diffusion.grid_spacing": "Finite-volume storage-axis spacing [m]",
                    "transport.numerical.cfl_advection_max": "Canonical operator advection limit [dimensionless]",
                    "transport.numerical.cfl_diffusion_max": "Canonical operator diffusion limit [dimensionless]",
                    "transport.numerical.default_boundary_type": "ParameterManager default boundary type; explicit experiment configuration may override it",
                    "transport.numerical.enforce_non_negative": "Non-negative state-domain guard [bool]",
                    "transport.numerical.track_conservation": "Conservation diagnostic flag [bool]",
                },
                input_variables=[
                    "tracer_concentration",
                    "source_term",
                    "velocity_field",
                    "tracer_concentration_boundary_state",
                ],
                output_variables=[
                    "tracer_concentration",
                    "tracer_concentration_advection_diffusion_boundary_flux",
                    "diagnostics.cfl_number",
                    "diagnostics.diffusion_stability",
                    "diagnostics.peclet_number",
                    "diagnostics.conservation_error",
                ],
                optional_output_variables=[
                    "tracer_concentration_advection_diffusion_boundary_flux",
                    "diagnostics.peclet_number",
                    "diagnostics.conservation_error",
                ],
                conservation_laws=["mass"],
                spatial_dimensions=spatial_dims,
                temporal_scale=TemporalScale.SECOND,
                coupling_interfaces=[],
                spatial_scale=SpatialScale.GRID_CELL,
                variable_kind=VariableKind.STATE,
            )
        super().__init__(name=functor_name, config=config or {}, metadata=metadata)
        self.boundary_state_key = "tracer_concentration_boundary_state"
        self.boundary_flux_key = (
            "tracer_concentration_advection_diffusion_boundary_flux"
        )
        self.optional_input_variables = {"source_term", self.boundary_state_key}

        # Set device
        self.device = device if device is not None else (torch.device("cpu"))

        # Resolve baseline diffusivity through the parameter manager when not supplied
        resolved_diffusivity = diffusivity
        if resolved_diffusivity is None:
            resolved_diffusivity = self.get_parameter_value(
                "diffusivity",
                namespace="transport.diffusion", required=True)
        if splitting_method is None:
            splitting_method = str(
                self.get_parameter_value(
                    "splitting_method",
                    namespace="transport.advection_diffusion",
                    required=True,
                )
            )

        # References: Hundsdorfer and Verwer (2003), Ch. I (advection-diffusion).
        # Mathematical expression: c_t + div(v c) = div(D grad c) + S with constant D.
        # Parameter meanings: D the constant diffusivity of this Functor.
        # Units: D [m2 s-1].
        # Domain/constraints: D finite and >= 0; D = 0 is a pure-advection step, for
        #   which the TransportMechanism diffusivity (defined only for D > 0) stays
        #   unset; any other value raises.
        # Sign convention: D >= 0; the diffusive flux points down the gradient.
        self.diffusivity_value = float(resolved_diffusivity)
        if not math.isfinite(self.diffusivity_value) or self.diffusivity_value < 0.0:
            raise ValueError(
                "AdvectionDiffusion diffusivity must be finite and >= 0 [m2 s-1], "
                f"got {self.diffusivity_value}"
            )
        if self.diffusivity_value > 0.0:
            self.set_scalar_diffusivity(self.diffusivity_value)

        # Validate inputs
        if spatial_dims not in [1, 2, 3]:
            raise ValueError("Spatial dimensions must be 1, 2, or 3")
        if splitting_method not in ["sequential", "symmetric", "combined"]:
            raise ValueError("Splitting method must be 'sequential', 'symmetric', or 'combined'")

        # Store configuration
        self.spatial_dims = spatial_dims
        self.splitting_method = splitting_method
        self._validate_configured_velocity(velocity_field)
        self.velocity_field_value = velocity_field
        if velocity_field is not None:
            # A configured velocity field replaces the state input, which becomes
            # optional under the required-input rule (FunctorBase.get_required_inputs);
            # without it the state input stays required and its absence raises.
            self.optional_input_variables = set(self.optional_input_variables) | {"velocity_field"}
        self.grid_spacing_value = grid_spacing
        self.advection_scheme_value = advection_scheme

        # Get parameters from parameter system
        if non_negative is None:
            non_negative = self.get_parameter_value(
                "enforce_non_negative",
                namespace="transport.numerical", required=True)
        self.non_negative = bool(non_negative)

        if grid_spacing is None:
            grid_spacing = self.get_parameter_value(
                "grid_spacing",
                namespace="transport.advection_diffusion", required=True)
        self.grid_spacing_value = self._normalize_grid_spacing(grid_spacing)

        track_conservation = self.get_parameter_value(
            "track_conservation",
            namespace="transport.numerical", required=True)
        self.track_conservation = bool(track_conservation)
        self.track_peclet = bool(track_peclet)
        # References: Patankar (1980), Numerical Heat Transfer and Fluid Flow,
        #   Sec. 5.2 (cell Peclet number); Hundsdorfer and Verwer (2003).
        # Mathematical expression: Pe_max = max_i,k |u_i,k| * dx_k / D.
        # Parameter meanings: u_i,k velocity component k in cell i, dx_k the
        #   cell width along axis k, D the diffusivity of this Functor.
        # Units: u [m s-1]; dx [m]; D [m2 s-1]; Pe [1].
        # Domain/constraints: Pe is defined only for D > 0; at D = 0 the ratio
        #   is 0/0 for u = 0 and unbounded otherwise, so tracking it is rejected
        #   here instead of inside a step; with track_peclet off a zero D is a
        #   valid pure-advection step.
        # Sign convention: Pe is a magnitude, independent of flow direction.
        if self.track_peclet and not self.diffusivity_value > 0.0:
            raise ValueError(
                "track_peclet requires a positive diffusivity: the grid Peclet "
                f"number Pe = |u| dx / D is undefined for D = {self.diffusivity_value}; "
                "use track_peclet=False for pure advection."
            )

        # Resolve one boundary method for the canonical coupled operator.
        if boundary_method is None:
            boundary_method = str(
                self.get_parameter_value(
                    "default_boundary_type",
                    namespace="transport.numerical",
                    required=True,
                )
            )
        self.boundary_method = str(boundary_method).strip().lower()
        supported_boundary_methods = {
            "periodic",
            "neumann",
            "dirichlet",
            "zero",
            "open",
            "extrapolate",
        }
        if self.boundary_method not in supported_boundary_methods:
            raise ValueError(
                "AdvectionDiffusion boundary_method must be one of "
                f"{sorted(supported_boundary_methods)}"
            )
        # Initialize operator factory
        self.operator_factory = OperatorFactory()

    def declared_forms(
        self,
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for advection-diffusion.

        References:
          - Hundsdorfer and Verwer (2003), advection-diffusion splitting.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          c_dot = -D_h(q_adv_h + q_diff_h) + G_h u_h.
        Parameter meanings:
          c_h is tracer concentration, v_h is velocity, d_h c_h is the tracer
          gradient, and q_h is the total boundary-oriented tracer flux.
        """

        n_dofs = self._structural_n_dofs(
            n_dofs=n_dofs,
            complex_bundle=complex_bundle,
        )
        common = {
            "complex_id": "transport_advection_diffusion",
            "n_dofs": n_dofs,
            "physical_domain": "transport",
            "orientation_convention": "outward_positive",
        }
        return (
            CochainSpaceSpec(
                degree=0,
                primality="primal",
                units="state_unit",
                extensivity="intensive",
                semantic_name="tracer_concentration",
                **common,
            ),
            CochainSpaceSpec(
                degree=1,
                primality="primal",
                units="m s-1",
                extensivity="intensive",
                semantic_name="velocity_field",
                **common,
            ),
            CochainSpaceSpec(
                degree=1,
                primality="primal",
                units="state_unit m-1",
                extensivity="intensive",
                semantic_name="tracer_gradient",
                **common,
            ),
            CochainSpaceSpec(
                degree=2,
                primality="dual",
                units="state_unit m s-1",
                extensivity="integral",
                semantic_name="total_transport_flux",
                boundary_role="boundary",
                **common,
            ),
        )

    @staticmethod
    def _structural_n_dofs(
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> int:
        """Resolve a positive structural audit size without a child Functor."""

        value = (
            int(n_dofs)
            if n_dofs is not None
            else int(getattr(complex_bundle, "n_dofs", 1))
        )
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive")
        return value

    def _structural_velocity_magnitude(self) -> float:
        """Return configured speed or the ParameterManager audit magnitude."""

        if self.velocity_field_value is None:
            return float(
                self.get_parameter_value(
                    "default_velocity_magnitude",
                    namespace="transport.advection",
                    required=True,
                )
            )
        components = (
            tuple(self.velocity_field_value)
            if isinstance(self.velocity_field_value, (tuple, list))
            else (self.velocity_field_value,)
        )
        squared = sum(torch.as_tensor(component, device=self.device) ** 2 for component in components)
        return float(torch.max(torch.sqrt(squared)).detach().to(dtype=torch.float64))

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return tuple(sorted({
            "transport.advection.default_velocity_magnitude",
            "transport.diffusion.diffusivity",
            "transport.advection_diffusion.grid_spacing",
            "transport.advection_diffusion.splitting_method",
            "transport.numerical.enforce_non_negative",
            "transport.numerical.track_conservation",
            "transport.numerical.default_boundary_type",
            "transport.numerical.cfl_advection_max",
            "transport.numerical.cfl_diffusion_max",
        }))

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build advection and diffusion material maps without child Functors."""

        n_dofs = self._structural_n_dofs(complex_bundle=complex_bundle)
        speed = max(self._structural_velocity_magnitude(), 0.0)
        diffusivity = float(
            (resolved_parameters or {}).get(
                "transport.diffusion.diffusivity",
                self.diffusivity_value,
            )
        )
        if diffusivity <= 0.0:
            raise ValueError("Diffusivity must be positive for Hodge assembly")
        forms = {
            form.semantic_name: form
            for form in self.declared_forms(complex_bundle=complex_bundle)
        }
        advective = HodgeStarSpec(
            name=f"{self.name}_advective_hodge",
            domain=forms["tracer_concentration"],
            codomain=forms["total_transport_flux"],
            matrix=torch.diag(
                torch.full((n_dofs,), speed, dtype=torch.float32, device=self.device)
            ),
            parameter_keys=("transport.advection.default_velocity_magnitude",),
            metric_keys=("transport.advection_diffusion.grid_spacing",),
            material_law="linear_advection_flux_magnitude",
        )
        diffusive = HodgeStarSpec(
            name=f"{self.name}_diffusive_hodge",
            domain=forms["tracer_gradient"],
            codomain=forms["total_transport_flux"],
            matrix=torch.diag(
                torch.full(
                    (n_dofs,),
                    diffusivity,
                    dtype=torch.float32,
                    device=self.device,
                )
            ),
            parameter_keys=("transport.diffusion.diffusivity",),
            metric_keys=("transport.advection_diffusion.grid_spacing",),
            material_law="fickian_diffusion",
        )
        advective.validate(passive=True)
        diffusive.validate(passive=True)
        return (advective, diffusive)

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
    ) -> Mapping[str, Any]:
        """Return the skew conservative advection contribution."""

        n_dofs = self._structural_n_dofs(complex_bundle=complex_bundle)
        matrix = torch.zeros((n_dofs, n_dofs), device=self.device)
        return {
            "state_vars": tuple(f"tracer_concentration[{idx}]" for idx in range(n_dofs)),
            "J": matrix - matrix.transpose(-1, -2),
        }

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return a PSD combined mobility contribution.

        References:
          - Hundsdorfer and Verwer (2003), split advection-diffusion.
          - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        Mathematical expression:
          M_h = M_adv_h + M_diff_h, with each term PSD.
        Parameter meanings:
          M_adv_h is zero for conservative advection; M_diff_h uses diffusivity.
        """

        n_dofs = self._structural_n_dofs(complex_bundle=complex_bundle)
        matrix = torch.diag(
            torch.full(
                (n_dofs,),
                float(self.diffusivity_value),
                dtype=torch.float32,
                device=self.device,
            )
        )
        return {
            "state_vars": tuple(f"tracer_concentration[{idx}]" for idx in range(n_dofs)),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the retained total boundary transport flow variable."""

        effort_space = CochainSpaceSpec(
            complex_id="transport_advection_diffusion",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="state_unit",
            extensivity="intensive",
            physical_domain="transport",
            semantic_name="boundary_tracer_state",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="transport_advection_diffusion",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="state_unit m s-1",
            extensivity="integral",
            physical_domain="transport",
            semantic_name="boundary_transport_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="advection_diffusion_boundary_flux",
                effort_var="tracer_concentration_boundary_state",
                flow_var=(
                    "tracer_concentration_advection_diffusion_boundary_flux"
                ),
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="state_unit2 m s-1",
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

    def forward(
        # References: Hundsdorfer and Verwer (2003); Morton and Mayers (2005); LeVeque (2007).
        # Mathematical expression: c_dot = -D_h (q_adv_h + q_diff_h) + G_h u_h;
        #   q_adv_h = i_v c_h; q_diff_h = -star_K d_h c_h; G_h u_h is source_term plus the
        #   routed open boundary flux; the step uses the configured operator splitting.
        # Parameter meanings: c tracer concentration, v velocity, star_K the diffusivity Hodge,
        #   D_h the finite-volume divergence, u_h the source and boundary ports.
        # Units: c [kg m-3]; v [m s-1]; K [m2 s-1]; source [kg m-3 s-1]; boundary flux
        #   [kg m-2 s-1]; dt [s].
        # Domain/constraints: finite fields on the declared grid; the combined advection-diffusion
        #   stability is checked by the unified operators; increments are per-step ACCUMULATE items.
        # Sign convention: outward boundary fluxes positive; sources positive into the tracer.
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any
    ) -> FluxDict:
        """
        Forward time step for advection-diffusion equation.

        Uses unified operators from operators.py to advance the advection-diffusion
        equation with the specified splitting method.

        Parameters:
            state : StateDict
                Current state variables
                Required keys: "tracer_concentration" [kg m-3]
                Optional keys: "source_term" [kg m-3 s-1], velocity components
            time : float
                Current time [s]
            dt : float
                Time step [s]
            **kwargs : Additional parameters

        Returns:
            increments : FluxDict
                Concentration increments and diagnostics
                Keys: "tracer_concentration", plus diagnostics
        """
        if (not math.isfinite(float(dt))) or float(dt) <= 0.0:
            raise ValueError("AdvectionDiffusion dt must be positive finite")
        state_dict = dict(state)
        concentration, squeeze_dims = self._resolve_forward_concentration(state_dict)
        self._require_finite_field(concentration, "tracer_concentration")
        source_term = self._prepare_source_term(state_dict.get("source_term"), concentration)
        self._require_finite_field(source_term, "source_term")
        boundary_state = self._prepare_boundary_state(
            state_dict.get(self.boundary_state_key),
            concentration,
        )
        velocity = self._prepare_forward_velocity(state_dict, concentration)
        self._require_finite_field(velocity, "velocity_field")
        grid_spacing = self._coerce_grid_spacing(self._prepare_grid_spacing(concentration.shape))

        result = self._compute_iphs_advection_diffusion_update(
            concentration=concentration,
            velocity=velocity,
            source_term=source_term,
            boundary_state=boundary_state,
            grid_spacing=grid_spacing,
            time_step=float(dt),
        )
        concentration_new = result["concentration"]
        diagnostics = self._build_forward_diagnostics(result=result)

        concentration, concentration_new = self._squeeze_forward_fields(
            concentration=concentration,
            concentration_new=concentration_new,
            squeeze_dims=squeeze_dims,
        )

        # References: Hundsdorfer and Verwer (2003); Strang (1968).
        # Mathematical expression: delta_c = [-div(v*c) + div(K*grad(c)) + S] * dt.
        # Parameter meanings: c [state unit], v [m s-1], K [m2 s-1], S [state unit s-1].
        # Open-system route: S is the explicitly declared optional source port.
        delta_concentration = concentration_new - concentration
        increments: FluxDict = {
            "tracer_concentration": self.wrap_update(
                delta_concentration,
                mode=UpdateMode.ACCUMULATE,
                per_timestep=True,
            )
        }
        boundary_fluxes = result.get("boundary_fluxes")
        if boundary_fluxes:
            # References: Patankar (1980); LeVeque (2002).
            # Mathematical expression: Delta M - dt*integral(S)dV
            #   + dt*sum_f(j_out,f*A_f) = 0.
            # Parameter meanings: j_out is the outward-positive sum of
            #   advective and diffusive face fluxes [kg m-2 s-1].
            # Open-system route: the computation-layer packer retains every
            #   face in deterministic order without merging batch ledgers.
            boundary_cochain = pack_outward_boundary_flux_cochain(
                boundary_fluxes,
                squeeze_batch=bool(squeeze_dims),
            )
            increments[self.boundary_flux_key] = UpdateItem(
                value=boundary_cochain,
                mode=UpdateMode.REPLACE,
                unit="kg m-2 s-1",
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            )
        self._attach_diagnostics(increments, diagnostics)
        return increments

    def _resolve_forward_concentration(self, state_dict: StateDict) -> Tuple[Tensor, int]:
        tracer_concentration = state_dict.get("tracer_concentration")
        if tracer_concentration is None:
            raise ValueError("State must contain canonical 'tracer_concentration'")
        concentration = tracer_concentration.to(self.device)
        return self._ensure_batch_dimension(concentration)

    @staticmethod
    def _require_finite_field(
        value: Union[Tensor, Tuple[Tensor, ...], float, Tuple[float, ...]],
        name: str,
    ) -> None:
        """Reject non-finite scientific inputs before operator evaluation."""
        values = value if isinstance(value, tuple) else (value,)
        for component in values:
            component_tensor = torch.as_tensor(component)
            if not torch.isfinite(component_tensor).all():
                raise ValueError(f"AdvectionDiffusion {name} must contain only finite values")

    def _prepare_forward_velocity(
        self,
        state_dict: StateDict,
        concentration: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        velocity = self._get_velocity_field(state_dict)
        if self.spatial_dims <= 1:
            return self._align_velocity_to_concentration(velocity, concentration)
        velocity = self._coerce_velocity_tuple(velocity, concentration)
        return self._align_velocity_to_concentration(velocity, concentration)

    def _coerce_velocity_tuple(
        self,
        velocity: Union[Tensor, Tuple[Tensor, ...], float, Tuple[float, ...]],
        concentration: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, ...], float, Tuple[float, ...]]:
        if isinstance(velocity, (int, float)):
            raise ValueError(
                f"AdvectionDiffusion requires {self.spatial_dims} velocity components"
            )
        if isinstance(velocity, torch.Tensor) and velocity.numel() == 1:
            raise ValueError(
                f"AdvectionDiffusion requires {self.spatial_dims} velocity components"
            )
        if isinstance(velocity, tuple) and velocity:
            if len(velocity) != self.spatial_dims:
                raise ValueError(
                    f"AdvectionDiffusion requires {self.spatial_dims} velocity components"
                )
            velocity_tuple: List[Tensor] = []
            for vel_comp in velocity:
                if isinstance(vel_comp, (int, float)):
                    velocity_tuple.append(
                        torch.tensor(vel_comp, device=self.device, dtype=concentration.dtype)
                    )
                else:
                    velocity_tuple.append(vel_comp.to(self.device, dtype=concentration.dtype))
            return tuple(velocity_tuple)
        return velocity

    def _validate_configured_velocity(self, velocity_field: Any) -> None:
        """Reject incomplete multidimensional velocity configuration."""
        if velocity_field is None or self.spatial_dims == 1:
            return
        if isinstance(velocity_field, Tensor):
            valid = velocity_field.dim() > 0 and velocity_field.shape[-1] == self.spatial_dims
        elif isinstance(velocity_field, (tuple, list)):
            valid = len(velocity_field) == self.spatial_dims
        else:
            valid = False
        if not valid:
            raise ValueError(
                f"AdvectionDiffusion requires {self.spatial_dims} velocity components"
            )

    @staticmethod
    def _coerce_grid_spacing(
        grid_spacing: Union[float, Tuple[float, ...], Tensor],
    ) -> Union[float, Tuple[float, ...]]:
        if isinstance(grid_spacing, torch.Tensor):
            if grid_spacing.numel() == 1:
                return float(grid_spacing.detach().reshape(-1)[0])
            return tuple(float(x) for x in grid_spacing.detach().reshape(-1).tolist())
        if isinstance(grid_spacing, tuple):
            return tuple(float(x) for x in grid_spacing)
        return float(grid_spacing)

    def _compute_iphs_advection_diffusion_update(
        self,
        *,
        concentration: Tensor,
        velocity: Union[Tensor, Tuple[Tensor, ...]],
        source_term: Tensor,
        boundary_state: Optional[Tensor],
        grid_spacing: Union[float, Tuple[float, ...]],
        time_step: float,
    ) -> Dict[str, Any]:
        """Compute the canonical IPHS/DEC advection-diffusion update.

        References:
          - Hundsdorfer and Verwer (2003), operator splitting.
          - Strang (1968), symmetric splitting.
          - LeVeque (2007), conservative finite differences.
        Mathematical expression:
          c_dot = -D_h(q_adv_h + q_diff_h) + G_h u_h,
          q_adv_h = i_v c_h, q_diff_h = -star_K d_h c_h.
        Parameter meanings:
          concentration: c_h state cochain.
          velocity: v_h advective velocity [m s-1].
          source_term: open input contribution in G_h u_h per unit time.
          time_step: scheduler-requested time step [s].
        """

        return self._run_forward_operator(
            concentration=concentration,
            velocity=velocity,
            source_term=source_term,
            boundary_state=boundary_state,
            grid_spacing=grid_spacing,
            time_step=time_step,
        )

    def _run_forward_operator(
        self,
        *,
        concentration: Tensor,
        velocity: Union[Tensor, Tuple[Tensor, ...]],
        source_term: Tensor,
        boundary_state: Optional[Tensor],
        grid_spacing: Union[float, Tuple[float, ...]],
        time_step: float,
    ) -> Dict[str, Any]:
        advection_diffusion_operator = self.operator_factory.create(
            operator_name="transport.advection_diffusion",
            name="advection_diffusion_operator",
            config={
                "diffusivity": float(self.diffusivity_value),
                "spatial_dims": self.spatial_dims,
                "splitting_method": self.splitting_method,
                "advection_scheme": self.advection_scheme_value,
                "boundary_method": self.boundary_method,
                "check_stability": True,
                "enforce_non_negative": self.non_negative,
            },
            device=self.device,
            dtype=concentration.dtype,
        )
        return cast(
            Dict[str, Any],
            advection_diffusion_operator.apply(
            concentration=concentration,
            velocity=velocity,
            diffusivity=self.diffusivity_value,
            grid_spacing=grid_spacing,
            time_step=time_step,
            spatial_dims=self.spatial_dims,
            splitting_method=self.splitting_method,
            advection_scheme=self.advection_scheme_value,
            boundary_method=self.boundary_method,
            check_stability=True,
            enforce_non_negative=self.non_negative,
            source_term=source_term,
            advection_boundary_state=(
                boundary_state
                if self.boundary_method in {"dirichlet", "open"}
                else None
            ),
            diffusion_boundary_state=(
                boundary_state if self.boundary_method == "dirichlet" else None
            ),
            check_conservation=(
                self.track_conservation and self.boundary_method == "periodic"
            ),
            ),
        )

    def _build_forward_diagnostics(
        self,
        *,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        cfl_number = result.get("cfl_number")
        stability_number = result.get("stability_number")
        if cfl_number is None or stability_number is None:
            raise RuntimeError(
                "Canonical advection-diffusion operator omitted stability diagnostics"
            )
        diagnostics: Dict[str, Any] = {
            "diagnostics.cfl_number": torch.max(
                torch.as_tensor(cfl_number, device=self.device)
            ),
            "diagnostics.diffusion_stability": torch.max(
                torch.as_tensor(stability_number, device=self.device)
            ),
        }
        conservation_error = result.get("conservation_error")
        if conservation_error is not None:
            diagnostics["diagnostics.conservation_error"] = conservation_error
        if self.track_peclet:
            peclet = result.get("peclet_number")
            if peclet is None:
                raise RuntimeError(
                    "Canonical advection-diffusion operator omitted peclet_number"
                )
            diagnostics["diagnostics.peclet_number"] = torch.max(
                torch.as_tensor(peclet, device=self.device)
            )
        return diagnostics

    @staticmethod
    def _squeeze_forward_fields(
        *,
        concentration: Tensor,
        concentration_new: Tensor,
        squeeze_dims: int,
    ) -> Tuple[Tensor, Tensor]:
        for _ in range(squeeze_dims):
            concentration = concentration.squeeze(0)
            concentration_new = concentration_new.squeeze(0)
        return concentration, concentration_new

    @staticmethod
    def _attach_diagnostics(
        increments: FluxDict,
        diagnostics: Dict[str, Any],
    ) -> None:
        for key, value in diagnostics.items():
            if value is not None:
                increments[key] = value

    def _velocity_from_state_field(
        self,
        state: StateDict,
    ) -> Optional[Union[Tensor, Tuple[Tensor, ...]]]:
        """Resolve velocity from packed state['velocity_field'] representation."""
        if "velocity_field" not in state:
            return None
        velocity_tensor = state["velocity_field"].to(self.device)
        if self.spatial_dims == 1:
            return velocity_tensor
        if velocity_tensor.dim() >= 1 and velocity_tensor.shape[-1] == self.spatial_dims:
            components = [velocity_tensor[..., idx] for idx in range(self.spatial_dims)]
            return components[0] if len(components) == 1 else tuple(components)
        raise ValueError(
            f"AdvectionDiffusion requires {self.spatial_dims} velocity components"
        )

    def _velocity_from_configured_default(self) -> Optional[Union[Tensor, Tuple[Tensor, ...]]]:
        """Resolve velocity from configured default velocity_field_value."""
        if self.velocity_field_value is None:
            return None
        velocity = self.velocity_field_value
        if isinstance(velocity, (int, float)):
            if self.spatial_dims == 1:
                return torch.tensor(velocity, device=self.device)
            raise ValueError(
                f"AdvectionDiffusion requires {self.spatial_dims} velocity components"
            )
        if isinstance(velocity, torch.Tensor) and velocity.numel() == 1:
            if self.spatial_dims == 1:
                return velocity
            raise ValueError(
                f"AdvectionDiffusion requires {self.spatial_dims} velocity components"
            )
        return velocity

    def _get_velocity_field(self, state: StateDict) -> Union[Tensor, Tuple[Tensor, ...]]:
        """
        Get velocity field from state dictionary or stored values.

        Parameters:
            state : StateDict
                Current state variables

        Returns:
            velocity : Union[Tensor, Tuple[Tensor, ...]]
                Velocity field(s) for advection
        """
        velocity_from_field = self._velocity_from_state_field(state)
        if velocity_from_field is not None:
            return velocity_from_field

        configured_velocity = self._velocity_from_configured_default()
        if configured_velocity is not None:
            return configured_velocity

        raise ValueError(
            "State must contain canonical 'velocity_field' unless an explicit "
            "velocity_field was configured at construction"
        )

    def _prepare_grid_spacing(self, concentration_shape: torch.Size) -> Union[float, Tuple[float, ...]]:
        """
        Prepare grid spacing for the given concentration shape.

        Parameters:
            concentration_shape : torch.Size
                Shape of the concentration tensor

        Returns:
            grid_spacing : Union[float, Tuple[float, ...]]
                Grid spacing(s) in each direction [m]
        """
        if self.grid_spacing_value is not None:
            return self.grid_spacing_value

        # Default: unit spacing
        if self.spatial_dims == 1:
            return 1.0
        elif self.spatial_dims == 2:
            return (1.0, 1.0)
        else:
            return (1.0, 1.0, 1.0)

    @staticmethod
    def _normalize_grid_spacing(grid_spacing: Optional[Union[float, Tuple[float, ...], Tensor]]) -> Optional[Union[float, Tuple[float, ...]]]:
        if grid_spacing is None:
            return None
        if isinstance(grid_spacing, Tensor):
            if grid_spacing.dim() == 0:
                return float(grid_spacing.detach().reshape(-1)[0])
            return tuple(float(value) for value in grid_spacing.detach().reshape(-1).tolist())
        if isinstance(grid_spacing, (list, tuple)):
            return tuple(float(value) for value in grid_spacing)
        return float(grid_spacing)

    def _prepare_source_term(
        self,
        value: Optional[Tensor],
        template: Tensor,
    ) -> Tensor:
        """Return the optional source port or the explicit no-source closure.

        References:
          - Hundsdorfer and Verwer (2003), advection-diffusion-reaction PDEs.
        Mathematical expression:
          S_h = 0 when the optional volumetric source port is absent.
        Parameter meanings:
          S_h is the tracer source rate [kg m-3 s-1].
        Open/closed system route:
          A supplied source is an open volumetric input; absence selects the
          declared closed no-source case rather than a parameter fallback.
        """
        if value is None:
            return torch.zeros_like(template, device=self.device)
        if not isinstance(value, torch.Tensor):
            raise TypeError("source_term must be a torch.Tensor")
        tensor = value.to(self.device, dtype=template.dtype)
        if tensor.shape != template.shape:
            try:
                tensor = torch.broadcast_to(tensor, template.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "source_term must be broadcast-compatible with "
                    "tracer_concentration"
                ) from exc
        return tensor

    def _prepare_boundary_state(
        self,
        value: Optional[Tensor],
        template: Tensor,
    ) -> Optional[Tensor]:
        """Validate the conditional external tracer state at domain faces.

        References:
          - Patankar (1980), finite-volume boundary fluxes.
          - LeVeque (2002), conservative finite-volume transport.
        Mathematical expression:
          j_adv,out = (v dot n)c_upwind and j_diff,out = -K grad(c) dot n.
        Parameter meanings:
          value is external tracer concentration [kg m-3], v is velocity
          [m s-1], K is diffusivity [m2 s-1], and n is the outward normal.
        Open/closed system route:
          Dirichlet and open advection require this port; closed boundaries
          reject it so an external reservoir cannot be silently discarded.
        """

        requires_state = self.boundary_method in {"dirichlet", "open"}
        if value is None:
            if requires_state:
                raise ValueError(
                    f"AdvectionDiffusion boundary '{self.boundary_method}' "
                    "requires explicit boundary_state"
                )
            return None
        if not requires_state:
            raise ValueError(
                "boundary_state is only supported for Dirichlet or open "
                "advection-diffusion"
            )
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"State '{self.boundary_state_key}' must be a torch.Tensor"
            )
        if value.is_complex():
            raise TypeError("AdvectionDiffusion boundary_state must be real")
        tensor = value.to(device=self.device, dtype=template.dtype)
        try:
            tensor = torch.broadcast_to(tensor, template.shape)
        except RuntimeError as exc:
            raise ValueError(
                "boundary_state must be broadcast-compatible with "
                "tracer_concentration"
            ) from exc
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(
                "AdvectionDiffusion boundary_state must contain only finite values"
            )
        if self.non_negative and bool(torch.any(tensor < 0.0)):
            raise ValueError("AdvectionDiffusion boundary_state must be non-negative")
        return tensor

    def _ensure_batch_dimension(self, field: Tensor) -> Tuple[Tensor, int]:
        squeeze_dims = 0
        field_out = field
        missing_dims = max(0, self.spatial_dims + 1 - field_out.dim())
        for _ in range(missing_dims):
            field_out = field_out.unsqueeze(0)
            squeeze_dims += 1
        return field_out, squeeze_dims

    def _align_velocity_to_concentration(
        self,
        velocity: Union[Tensor, Tuple[Tensor, ...], float, Tuple[float, ...]],
        concentration: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        if isinstance(velocity, list):
            velocity = tuple(velocity)
        if isinstance(velocity, tuple):
            aligned: List[Tensor] = []
            for component in velocity:
                if isinstance(component, Tensor):
                    tensor = component.to(self.device, dtype=concentration.dtype)
                else:
                    tensor = torch.as_tensor(component, device=self.device, dtype=concentration.dtype)
                if tensor.shape != concentration.shape:
                    tensor = torch.broadcast_to(tensor, concentration.shape)
                aligned.append(tensor)
            return tuple(aligned)

        if isinstance(velocity, Tensor):
            tensor = velocity.to(self.device, dtype=concentration.dtype)
        else:
            tensor = torch.as_tensor(velocity, device=self.device, dtype=concentration.dtype)
        if tensor.shape != concentration.shape:
            tensor = torch.broadcast_to(tensor, concentration.shape)
        return tensor

# Utility functions for common setups
def create_wind_transport(
    spatial_dims: int,
    wind_speed: float,
    wind_direction: Tuple[float, ...],
    molecular_diffusivity: float,
    turbulent_diffusivity: float,
    grid_spacing: GridSpacing,
    device: Optional[torch.device] = None
) -> AdvectionDiffusion:
    """
    Create advection-diffusion for atmospheric transport with wind.

    Parameters:
        spatial_dims : int
            Number of spatial dimensions
        wind_speed : float
            Wind speed magnitude [m s-1]
        wind_direction : Tuple[float, ...]
            Unit wind direction vector
        molecular_diffusivity : float
            Molecular diffusion coefficient [m^2 s-1]
        turbulent_diffusivity : float
            Turbulent diffusion coefficient [m^2 s-1]
        grid_spacing : GridSpacing
            Grid spacing(s) [m]
        device : Optional[torch.device]
            Computation device

    Returns:
        transport : AdvectionDiffusion
            Configured atmospheric transport mechanism
    """
    if device is None:
        device = torch.device("cpu")

    if not math.isfinite(float(wind_speed)) or float(wind_speed) < 0.0:
        raise ValueError("wind_speed must be finite and non-negative")
    if len(wind_direction) != spatial_dims:
        raise ValueError(f"Wind direction must have {spatial_dims} components")
    if not all(math.isfinite(float(component)) for component in wind_direction):
        raise ValueError("wind_direction must contain only finite components")
    direction_norm = math.sqrt(
        sum(float(component) ** 2 for component in wind_direction)
    )
    direction_tolerance = math.sqrt(math.ulp(1.0))
    if not math.isclose(
        direction_norm,
        1.0,
        rel_tol=direction_tolerance,
        abs_tol=direction_tolerance,
    ):
        raise ValueError("wind_direction must be a unit vector")
    if (
        not math.isfinite(float(molecular_diffusivity))
        or not math.isfinite(float(turbulent_diffusivity))
        or float(molecular_diffusivity) < 0.0
        or float(turbulent_diffusivity) < 0.0
    ):
        raise ValueError("wind diffusivities must be finite and non-negative")

    # References: Stull (1988), An Introduction to Boundary Layer Meteorology.
    # Mathematical expression: K_eff = K_molecular + K_turbulent; v = speed * direction.
    # Parameter meanings: K terms [m2 s-1], speed [m s-1], direction [dimensionless].
    # Open-system route: this factory configures transport; boundary exchange uses Functor ports.
    total_diffusivity = molecular_diffusivity + turbulent_diffusivity
    if total_diffusivity <= 0.0:
        raise ValueError("total wind diffusivity must be positive")

    velocity_components = []
    for i in range(spatial_dims):
        vel_component = torch.tensor(wind_speed * wind_direction[i], device=device)
        velocity_components.append(vel_component)

    velocity_field = velocity_components[0] if spatial_dims == 1 else tuple(velocity_components)

    return AdvectionDiffusion(
        spatial_dims=spatial_dims,
        diffusivity=total_diffusivity,
        velocity_field=velocity_field,
        grid_spacing=grid_spacing,
        device=device
    )


def create_groundwater_transport(
    spatial_dims: int,
    darcy_velocity: Union[float, Tuple[float, ...]],
    porosity: float,
    molecular_diffusivity: float,
    dispersivity: float,
    grid_spacing: GridSpacing,
    device: Optional[torch.device] = None
) -> AdvectionDiffusion:
    """
    Create advection-diffusion for groundwater solute transport.

    Parameters:
        spatial_dims : int
            Number of spatial dimensions
        darcy_velocity : Union[float, Tuple[float, ...]]
            Darcy velocity [m s-1]
        porosity : float
            Soil porosity [dimensionless]
        molecular_diffusivity : float
            Molecular diffusion coefficient [m^2 s-1]
        dispersivity : float
            Mechanical dispersivity [m]
        grid_spacing : GridSpacing
            Grid spacing(s) [m]
        device : Optional[torch.device]
            Computation device

    Returns:
        transport : AdvectionDiffusion
            Configured groundwater transport mechanism
    """
    if device is None:
        device = torch.device("cpu")

    if not math.isfinite(float(porosity)) or not (0.0 < float(porosity) <= 1.0):
        raise ValueError("porosity must be finite and in the interval (0, 1]")
    if isinstance(darcy_velocity, (int, float)):
        if spatial_dims != 1:
            raise ValueError(
                "multidimensional darcy_velocity must provide all spatial components"
            )
        darcy_components = (float(darcy_velocity),)
    else:
        darcy_components = tuple(float(component) for component in darcy_velocity)
        if len(darcy_components) != spatial_dims:
            raise ValueError(
                "multidimensional darcy_velocity must provide all spatial components"
            )
    if not all(math.isfinite(component) for component in darcy_components):
        raise ValueError("darcy_velocity must contain only finite components")
    if (
        not math.isfinite(float(molecular_diffusivity))
        or not math.isfinite(float(dispersivity))
        or float(molecular_diffusivity) < 0.0
        or float(dispersivity) < 0.0
    ):
        raise ValueError(
            "groundwater diffusivity and dispersivity must be finite and non-negative"
        )

    # References: Bear (1972), Dynamics of Fluids in Porous Media.
    # Mathematical expression: v_pore = q_Darcy / n_e; D = D_m + alpha_L * norm(v_pore).
    # Parameter meanings: q_Darcy [m s-1], n_e [1], D_m [m2 s-1], alpha_L [m].
    # Open-system route: this factory configures transport; boundary exchange uses Functor ports.
    pore_components = tuple(component / porosity for component in darcy_components)
    velocity_components = tuple(
        torch.tensor(component, device=device) for component in pore_components
    )
    velocity_field = velocity_components[0] if spatial_dims == 1 else velocity_components
    velocity_mag = math.sqrt(sum(component ** 2 for component in pore_components))
    dispersive_diffusivity = dispersivity * velocity_mag
    total_diffusivity = molecular_diffusivity + dispersive_diffusivity
    if total_diffusivity <= 0.0:
        raise ValueError("total groundwater diffusivity must be positive")

    return AdvectionDiffusion(
        spatial_dims=spatial_dims,
        diffusivity=total_diffusivity,
        velocity_field=velocity_field,
        grid_spacing=grid_spacing,
        device=device
    )


# Export main classes and functions
__all__ = [
    'AdvectionDiffusion',
    'create_wind_transport',
    'create_groundwater_transport',
]

