r"""Linear advection transport module for RELIC.

This module provides the dedicated linear advection Functor. Production
numerics are owned only by the unified computation-layer advection operator;
the Functor supplies lifecycle integration, boundary-port
semantics, explicit state updates, and conservation diagnostics.

Mathematical Expressions
------------------------
1. Linear advection equation: ``\partial c / \partial t + \nabla \cdot (\mathbf{v} c) = 0``
2. One-dimensional upwind flux: ``F_{i+1/2} = u_{i+1/2} c_{\text{upwind}}``
3. Outflow CFL: ``\Delta t \sum_k [\max(u_{k,+},0)-\min(u_{k,-},0)] / \Delta x_k``

References
----------
- Courant, R., Friedrichs, K., Lewy, H. (1928). "On the partial difference
  equations of mathematical physics." Mathematische Annalen 100: 32-74.
- LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary and Partial
  Differential Equations.* SIAM.
- Sweby, P.K. (1984). "High resolution schemes using flux limiters for
  hyperbolic conservation laws." SIAM Journal on Numerical Analysis, 21(5),
  995-1011.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from ...core.abstractions import FluxDict, StateDict
from ...core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from ...core.functor import TransportMechanismFunctor
from ...core.state import UpdateItem, UpdateMode
# The computation-layer operator is the single owner of scheme formulas.
from relic.computation.operators import OperatorFactory
from relic.computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from relic.computation.operators.functional import pack_outward_boundary_flux_cochain
from relic.computation.operators.iphs import PortSpecIPHS

SUPPORTED_SCHEMES = {"upwind", "lax_wendroff", "superbee", "van_leer"}
SUPPORTED_BOUNDARY_METHODS = {
    "periodic",
    "neumann",
    "dirichlet",
    "zero",
    "open",
    "extrapolate",
}


class VelocityFieldError(ValueError):
    """Raised when velocity field input is inconsistent with spatial dimensions."""


def advective_flux_units(tracer_units: str) -> str:
    """Return the unit label of the advective face flux for a tracer stored in ``tracer_units``.

    References:
      - LeVeque (2002), Finite Volume Methods for Hyperbolic Problems, face fluxes.
      - BIPM SI Brochure (9th ed.), Section 2.3.3, the unit one.
    Mathematical expression:
      j_out = (v . n) c, so [j_out] = [c] m s-1; for a density [c] = X m-3 this is X m-2 s-1.
    Parameter meanings:
      tracer_units is the declared unit label of the advected state c.
    Units:
      returns the label of j_out; kg m-3 -> kg m-2 s-1; dimensionless or 1 -> m s-1;
      any other label L -> "L m s-1".
    Domain/constraints:
      tracer_units must be a non-empty label; the mapping is a deterministic label rule,
      not a unit conversion, so callers must keep the state and the tracer label consistent.
    Sign convention:
      j_out is positive out of the domain.
    """
    label = str(tracer_units).strip()
    if not label:
        raise ValueError("tracer_units must be a non-empty unit label.")
    if label in {"dimensionless", "1"}:
        return "m s-1"
    if label.endswith(" m-3"):
        return f"{label[: -len(' m-3')]} m-2 s-1"
    return f"{label} m s-1"


class LinearAdvection(TransportMechanismFunctor):
    """Linear advection functor with rich diagnostics and parameter integration.

    The implementation follows the architectural requirements of the V2 framework:
    parameters are resolved exclusively via the parameter manager, metadata is
    documented through :class:`~relic.core.functor.FunctorMetadata`, and
    conservation diagnostics are routed through :class:`~relic.core.StateManager`.

    Parameters
    ----------
    spatial_dims : int, optional
        Number of spatial dimensions (1, 2, or 3). Units: dimensionless.
    velocity_field : Optional[Union[Tensor, Sequence[Tensor]]]
        Velocity components (``m s-1``). Sequence length must equal
        ``spatial_dims``. If omitted, each forward state must provide the
        canonical ``velocity_field`` input.
    grid_spacing : Optional[Union[float, Sequence[float]]]
        Grid spacing in metres in tensor storage order: ``(dy, dx)`` for 2D
        and ``(dz, dy, dx)`` for 3D.
    scheme : str, optional
        Numerical scheme identifier: ``"upwind"``, ``"lax_wendroff"``,
        ``"superbee"`` or ``"van_leer"``.
    boundary_method : Optional[str]
        Canonical operator boundary method. Explicit constructor values take
        precedence over the config ``boundary_method`` and ParameterManager
        ``transport.numerical.default_boundary_type``.
    non_negative : Optional[bool]
        Whether to enforce non-negative concentrations.
    track_conservation : Optional[bool]
        Enable conservation diagnostics using ``validation.conservation`` tolerances.
    device : Optional[torch.device]
        Target PyTorch device.

    Notes
    -----
    - Mass conservation is monitored through ``self.track_conservation``.
    - The unified finite-volume operator owns the CFL calculation and fails fast
      when the scheduler-requested step requires simulator-managed substeps.
    """

    TRANSPORT_TYPE = "advection"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"

    def __init__(
        self,
        spatial_dims: int = 1,
        velocity_field: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        grid_spacing: Optional[Union[float, Sequence[float]]] = None,
        tracer_key: str = "tracer_concentration",
        tracer_units: str = "kg m-3",
        scheme: str = "upwind",
        boundary_method: Optional[str] = None,
        non_negative: Optional[bool] = None,
        track_conservation: Optional[bool] = None,
        device: Optional[torch.device] = None,
        name: str = "linear_advection",
        config: Optional[Dict[str, Union[str, float, int, bool]]] = None,
    ) -> None:
        self._validate_spatial_dims(spatial_dims)
        self._validate_scheme(scheme)

        metadata = self._create_metadata(name, tracer_key)
        transport_kwargs = {
            "device": device if device is not None else torch.device("cpu"),
        }
        super().__init__(name=name, config=config or {}, metadata=metadata, **transport_kwargs)
        self.boundary_state_key = f"{tracer_key}_boundary_state"
        self.boundary_flux_key = f"{tracer_key}_advective_boundary_flux"
        self.optional_input_variables = {
            "source_term",
            self.boundary_state_key,
        }

        self.device = device if device is not None else torch.device("cpu")
        self.spatial_dims = spatial_dims
        self.scheme = scheme
        self.tracer_key = tracer_key
        # The tracer state unit is a declared contract of the instance; the
        # advective boundary flux label follows from it (see advective_flux_units).
        self.boundary_flux_units = advective_flux_units(tracer_units)
        self.tracer_units = str(tracer_units).strip()
        self.boundary_method = self._resolve_boundary_method(boundary_method)
        
        self.operator_factory = OperatorFactory()
        self._advection_operator = None

        self.grid_spacing = self._initialize_grid_spacing(grid_spacing)
        self.velocity_field: Optional[Tuple[Tensor, ...]] = None
        if velocity_field is not None:
            self.set_velocity_field(velocity_field)

        numerical_params = self._load_numerical_parameters()

        if non_negative is None:
            non_negative = numerical_params["non_negative"]
        self.non_negative = bool(non_negative)

        if track_conservation is None:
            track_conservation = numerical_params["track_conservation"]
        self.track_conservation = bool(track_conservation)

        self._initialize_advection_operator(name)

    # ------------------------------------------------------------------
    # Metadata and parameter helpers
    # ------------------------------------------------------------------
    def _resolve_boundary_method(self, explicit: Optional[str]) -> str:
        """Resolve and validate the single operator boundary method."""

        selected: Any = explicit
        if selected is None:
            selected = self.get_parameter_value(
                "boundary_method",
                namespace="transport.advection",
                source_order=("config",),
                required=False,
            )
        if selected is None:
            selected = self.get_parameter_value(
                "default_boundary_type",
                namespace="transport.numerical",
                source_order=("manager",),
                required=True,
            )

        normalized = str(selected).strip().lower()
        if normalized not in SUPPORTED_BOUNDARY_METHODS:
            raise ValueError(
                "LinearAdvection boundary_method must be one of "
                f"{sorted(SUPPORTED_BOUNDARY_METHODS)}"
            )
        return normalized

    def _create_metadata(self, name: str, tracer_key: str) -> FunctorMetadata:
        """Build functor metadata with scientific references."""

        boundary_state_key = f"{tracer_key}_boundary_state"
        boundary_flux_key = f"{tracer_key}_advective_boundary_flux"

        parameters = {
            "transport.advection.grid_spacing": "Finite-volume grid spacing [m]",
            "transport.advection.default_velocity_magnitude": (
                "Structural Hodge audit velocity magnitude only [m s-1]"
            ),
            "transport.advection.inflow_boundary_type": (
                "Inflow boundary type for open advection [categorical]"
            ),
            "transport.advection.outflow_boundary_type": (
                "Outflow boundary type for open advection [categorical]"
            ),
            "transport.numerical.cfl_advection_max": "Maximum stabilising CFL number [dimensionless]",
            "transport.numerical.track_conservation": "Closed-domain conservation audit flag [bool]",
            "transport.numerical.enforce_non_negative": "Non-negative state-domain guard [bool]",
            "transport.numerical.default_boundary_type": "Default boundary method [categorical]",
        }

        return FunctorMetadata(
            name=name,
            version="1.0.0",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Linear advection transport mechanism with CFL diagnostics.",
            mathematical_expression="c_dot = -D_h q_h + s_h; q_h = i_v c_h",
            references=[
                "Courant et al. (1928) Mathematische Annalen 100:32-74",
                "LeVeque (2007) Finite Difference Methods for ODE/PDE",
                "Sweby (1984) SIAM J. Numer. Anal. 21(5):995-1011",
            ],
            conservation_laws=["mass"],
            parameters=parameters,
            input_variables=[
                tracer_key,
                "velocity_field",
                "source_term",
                boundary_state_key,
            ],
            output_variables=[
                tracer_key,
                boundary_flux_key,
                "diagnostics.cfl_number",
                "diagnostics.conservation_error",
            ],
            optional_output_variables=[
                boundary_flux_key,
                "diagnostics.conservation_error",
            ],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )

    def _load_numerical_parameters(self) -> Dict[str, Union[float, bool]]:
        """Resolve numerical control parameters via FunctorBase parameter chain."""
        parameter_specs: List[Tuple[str, str, Optional[str], type]] = [
            ("track_conservation", "track_conservation", None, bool),
            ("non_negative", "enforce_non_negative", None, bool),
        ]

        params: Dict[str, Union[float, bool]] = {}
        for attr, param_name, units, cast_type in parameter_specs:
            value = self.get_parameter_value(
                param_name,
                namespace="transport.numerical",
                units=units,
                expected_type=cast_type if cast_type is not bool else None,
                source_order=("manager",),
                required=True,
            )
            if cast_type is bool:
                params[attr] = bool(value)
            else:
                params[attr] = cast_type(value)

        return params  # type: ignore[return-value]

    def _structural_n_dofs(
        self,
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> int:
        """Resolve local DEC degrees of freedom for structural audits."""

        if n_dofs is not None:
            value = int(n_dofs)
        elif hasattr(complex_bundle, "n_dofs"):
            value = int(getattr(complex_bundle, "n_dofs"))
        else:
            value = 1
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for linear advection.

        References:
          - LeVeque (2007), finite-volume advection.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          c_dot = -D_h q_h + s_h, q_h = i_v c_h.
        Parameter meanings:
          n_dofs is a structural audit size; velocity is supplied by state or parameters.
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="transport_linear_advection",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="state_unit",
                extensivity="intensive",
                physical_domain="transport",
                semantic_name=self.tracer_key,
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="transport_linear_advection",
                degree=1,
                primality="primal",
                n_dofs=n_dofs,
                units="m s-1",
                extensivity="intensive",
                physical_domain="transport",
                semantic_name="velocity_field",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="transport_linear_advection",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="state_unit m s-1",
                extensivity="integral",
                physical_domain="transport",
                semantic_name="advective_tracer_flux",
                orientation_convention="outward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "transport.advection.grid_spacing",
            "transport.advection.default_velocity_magnitude",
            "transport.advection.inflow_boundary_type",
            "transport.advection.outflow_boundary_type",
            "transport.numerical.cfl_advection_max",
            "transport.numerical.track_conservation",
            "transport.numerical.enforce_non_negative",
            "transport.numerical.default_boundary_type",
        )

    def _structural_velocity_magnitude(self) -> float:
        """Return the maximum Euclidean speed for the structural Hodge view.

        References: LeVeque (2007); Hirani (2003).
        Mathematical expression: speed = max_x sqrt(sum_i(v_i(x)^2)).
        Parameter meanings: v_i is velocity component i [m s-1].
        """

        if self.velocity_field is not None:
            speed = self._compute_velocity_magnitude().detach()
            return float(torch.max(speed).to(dtype=torch.float64))
        value = self.get_parameter_value(
            "default_velocity_magnitude",
            namespace="transport.advection",
            required=True,
        )
        return float(value)

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the advective magnitude Hodge contract.

        References:
          - LeVeque (2007), conservative advection fluxes.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          q_h = sign(v_h) star_|v| c_h; star_|v| = diag(|v_i|) >= 0.
        Parameter meanings:
          default_velocity_magnitude supplies |v| when no state velocity is set.
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        speed = max(self._structural_velocity_magnitude(), 0.0)
        values = torch.full((n_dofs,), speed, dtype=torch.float32, device=self.device)
        domain = CochainSpaceSpec(
            complex_id="transport_linear_advection",
            degree=0,
            primality="primal",
            n_dofs=n_dofs,
            units="state_unit",
            extensivity="intensive",
            physical_domain="transport",
            semantic_name=self.tracer_key,
            orientation_convention="outward_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="transport_linear_advection",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="state_unit m s-1",
            extensivity="integral",
            physical_domain="transport",
            semantic_name="advective_tracer_flux",
            orientation_convention="outward_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=("transport.advection.default_velocity_magnitude",),
            metric_keys=("transport.advection.grid_spacing",),
            material_law="linear_advection_flux_magnitude",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build advection magnitude Hodge terms for IPHS assembly."""

        del resolved_parameters
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
    ) -> Mapping[str, Any]:
        """Return the conservative advection interconnection view."""

        n_dofs = self._structural_n_dofs(complex_bundle=complex_bundle)
        matrix = torch.zeros((n_dofs, n_dofs), device=self.device, dtype=torch.float32)
        return {
            "state_vars": tuple(f"{self.tracer_key}[{idx}]" for idx in range(n_dofs)),
            "J": matrix - matrix.transpose(-1, -2),
        }

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return a PSD zero dissipation matrix for conservative advection.

        References:
          - LeVeque (2007), conservative advection.
          - van der Schaft and Maschke (1995), port-Hamiltonian structure.
        Mathematical expression:
          M_h = 0, sigma = X_h.T M_h X_h = 0 >= 0.
        Parameter meanings:
          n_dofs is the structural audit size.
        """

        n_dofs = self._structural_n_dofs(complex_bundle=complex_bundle)
        matrix = torch.zeros((n_dofs, n_dofs), device=self.device, dtype=torch.float32)
        return {
            "state_vars": tuple(f"{self.tracer_key}[{idx}]" for idx in range(n_dofs)),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the advective boundary flux port."""

        effort_space = CochainSpaceSpec(
            complex_id="transport_linear_advection",
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
            complex_id="transport_linear_advection",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="state_unit m s-1",
            extensivity="integral",
            physical_domain="transport",
            semantic_name="boundary_advective_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="advective_boundary_flux",
                effort_var=f"{self.tracer_key}_boundary_state",
                flow_var=f"{self.tracer_key}_advective_boundary_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="state_unit2 m s-1",
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

    def _initialize_advection_operator(self, name: str) -> None:
        self._advection_operator = self.operator_factory.get_operator(
            "transport.advection",
            name=f"{name}_operator",
            config={
                "spatial_dims": self.spatial_dims,
                "scheme": self.scheme,
                "boundary_method": self.boundary_method,
                "check_cfl": True,
            },
            device=self.device,
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_spatial_dims(spatial_dims: int) -> None:
        if spatial_dims not in (1, 2, 3):
            raise ValueError("Spatial dimensions must be 1, 2, or 3")

    @staticmethod
    def _validate_scheme(scheme: str) -> None:
        if scheme not in SUPPORTED_SCHEMES:
            raise ValueError("Scheme must be one of " + ", ".join(sorted(SUPPORTED_SCHEMES)))

    # ------------------------------------------------------------------
    # Grid and velocity setup
    # ------------------------------------------------------------------
    def _initialize_grid_spacing(
        self, grid_spacing: Optional[Union[float, Sequence[float]]]
    ) -> List[float]:
        if grid_spacing is None:
            default_spacing = self.get_parameter_value(
                "grid_spacing",
                namespace="transport.advection",
                source_order=("manager",),
            )
            if default_spacing is None:
                raise KeyError("Missing required parameter 'transport.advection.grid_spacing'")
            spacing = [float(default_spacing)] * self.spatial_dims
        elif isinstance(grid_spacing, (int, float)):
            spacing = [float(grid_spacing)] * self.spatial_dims
        else:
            spacing = [float(value) for value in grid_spacing]
            if len(spacing) != self.spatial_dims:
                raise ValueError(
                    f"Grid spacing must have {self.spatial_dims} values, got {len(spacing)}"
                )

        if any(not math.isfinite(value) or value <= 0.0 for value in spacing):
            raise ValueError("Grid spacing values must be finite and positive")
        return spacing

    def set_velocity_field(
        self, velocity_field: Union[Tensor, Sequence[Tensor], Sequence[float]]
    ) -> None:
        """Set velocity field components ensuring dimensional consistency."""

        components = self._normalize_velocity_input(velocity_field)
        if len(components) != self.spatial_dims:
            raise VelocityFieldError(
                f"Need {self.spatial_dims} velocity components, received {len(components)}"
            )

        tensors: List[Tensor] = []
        for comp in components:
            tensor = comp if isinstance(comp, Tensor) else torch.as_tensor(comp)
            tensors.append(tensor.to(self.device, copy=True))
        self.velocity_field = tuple(tensors)
        # A bound velocity field replaces the state input, which becomes optional
        # under the required-input rule (FunctorBase.get_required_inputs).
        self.optional_input_variables = set(self.optional_input_variables) | {"velocity_field"}

    def _normalize_velocity_input(
        self, velocity_field: Union[Tensor, Sequence[Tensor], Sequence[float]]
    ) -> List[Tensor]:
        if isinstance(velocity_field, Tensor):
            if velocity_field.dim() > 0 and velocity_field.shape[-1] == self.spatial_dims:
                return [velocity_field[..., idx] for idx in range(self.spatial_dims)]
            if self.spatial_dims == 1:
                return [velocity_field]
            raise VelocityFieldError(
                f"Need {self.spatial_dims} velocity components"
            )

        components: List[Tensor] = []
        for value in velocity_field:  # type: ignore[arg-type]
            if isinstance(value, Tensor):
                components.append(value)
            else:
                components.append(torch.tensor(float(value)))
        return components

    # ------------------------------------------------------------------
    # Structural diagnostics utilities
    # ------------------------------------------------------------------
    def _compute_velocity_magnitude(
        self,
        velocity_field: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None,
    ) -> Tensor:
        """Compute pointwise Euclidean speed from velocity components.

        References: LeVeque (2007), advective transport speed.
        Mathematical expression: speed(x) = sqrt(sum_i(v_i(x)^2)).
        Parameter meanings: v_i is velocity component i [m s-1].
        """

        field = velocity_field if velocity_field is not None else self.velocity_field
        if field is None:
            raise RuntimeError("Velocity field not set")
        if isinstance(field, tuple):
            squared = sum(comp.to(self.device) ** 2 for comp in field)
        else:
            squared = field.to(self.device) ** 2
        return torch.sqrt(squared)

    # ------------------------------------------------------------------
    # Core functor interfaces
    # ------------------------------------------------------------------
    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Tensor) -> FluxDict:
        # References: Courant et al. (1928); LeVeque (2007); Sweby (1984).
        # Mathematical expression: c_dot = -D_h q_h + s_h; q_h = i_v c_h with the selected upwind,
        #   Lax-Wendroff or flux-limited face scheme; CFL = max|v| dt / dx; Delta c = c_dot dt;
        #   open-boundary faces route j_out = (v . n) c_upwind.
        # Parameter meanings: c the tracer state in tracer_units, v the velocity field, s_h the
        #   optional source term, D_h the finite-volume divergence, j_out the outward face flux.
        # Units: c [tracer_units, kg m-3 by default]; v [m s-1]; s_h [tracer_units s-1];
        #   j_out [tracer_units x m s-1]; dt [s]; CFL [1].
        # Domain/constraints: finite tracer and velocity fields on the declared grid; the CFL
        #   diagnostic is reported and an unstable direct step fails; increments are ACCUMULATE
        #   per step and the boundary flux is a REPLACE port.
        # Sign convention: j_out positive out of the domain; sources positive into the tracer.
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("Time step must be finite and positive")

        concentration = self._extract_concentration_from_state(state)
        if self.non_negative and bool(torch.any(concentration < 0.0)):
            raise ValueError(
                "LinearAdvection requires a non-negative concentration state; "
                "negative inventories cannot be repaired by mass-creating clipping"
            )

        boundary_state = state.get(self.boundary_state_key)
        requires_boundary_state = self.boundary_method in {"dirichlet", "open"}
        if requires_boundary_state and boundary_state is None:
            raise ValueError(
                f"Advection boundary '{self.boundary_method}' requires explicit "
                "boundary_state"
            )
        if boundary_state is not None and not requires_boundary_state:
            raise ValueError(
                "boundary_state is only supported for Dirichlet or open advection"
            )
        if boundary_state is not None:
            if not isinstance(boundary_state, Tensor):
                raise TypeError(f"State '{self.boundary_state_key}' must be a torch.Tensor")
            if not bool(torch.all(torch.isfinite(boundary_state))):
                raise ValueError(
                    f"State '{self.boundary_state_key}' must contain only finite values"
                )
            if self.non_negative and bool(torch.any(boundary_state < 0.0)):
                raise ValueError("Advection boundary_state must be non-negative")

        velocity = self._prepare_velocity_for_operator(state)
        concentration_op, velocity, squeeze_dims = self._prepare_operator_inputs(
            concentration,
            velocity,
        )

        update = self._compute_iphs_advection_update(
            concentration=concentration,
            concentration_op=concentration_op,
            velocity=velocity,
            dt=dt,
            boundary_state=boundary_state,
        )
        dt_effective = float(update["dt_effective"])
        result = update["operator_result"]

        concentration_new = result.get("concentration")
        if concentration_new is None:
            raise RuntimeError("Advection operator did not return concentration")
        for _ in range(squeeze_dims):
            concentration_new = concentration_new.squeeze(0)
        concentration_new = concentration_new.to(
            device=concentration.device,
            dtype=concentration.dtype,
        )
        # References: LeVeque (2002), finite-volume state advancement.
        # Expression: Delta c = c^(n+1) - c^n.
        # Parameters: c is the advected tracer state [tracer_units, kg m-3 for the canonical tracer].
        # Open route: boundary exchange remains in the operator face ledger.
        delta_concentration = concentration_new - concentration
        increments: FluxDict = {}

        cfl_tensor = result.get("cfl_number")
        if cfl_tensor is None:
            raise RuntimeError("Advection operator did not return cfl_number")
        diagnostics_cfl = torch.max(cfl_tensor)

        increments["diagnostics.cfl_number"] = diagnostics_cfl

        if "source_term" in state:
            source_term = state["source_term"]
        else:
            source_term = None
        if source_term is not None:
            if not isinstance(source_term, Tensor):
                raise TypeError("source_term must be a torch.Tensor")
            if not bool(torch.all(torch.isfinite(source_term))):
                raise ValueError("source_term must contain only finite values")
            try:
                torch.broadcast_shapes(source_term.shape, concentration.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "source_term must be broadcast-compatible with tracer_concentration"
                ) from exc
            # References: Patankar (1980); LeVeque (2007).
            # Expression: delta_c_source = S * dt.
            # Parameters: S [concentration s-1], dt [s].
            dt_tensor = torch.as_tensor(dt_effective, device=self.device, dtype=concentration.dtype)
            source_increment = source_term.to(self.device, dtype=concentration.dtype) * dt_tensor
            delta_concentration = delta_concentration + source_increment

        concentration_new = concentration + delta_concentration
        if self.non_negative and bool(torch.any(concentration_new < 0.0)):
            minimum = float(torch.min(concentration_new.detach()))
            raise ValueError(
                "LinearAdvection step violated non-negative concentration: "
                f"minimum={minimum:.6g}. Reduce time_step or use existing "
                "simulator-managed substeps."
            )

        if self.track_conservation:
            conservation = result.get("conservation_error")
            if conservation is None:
                raise RuntimeError(
                    "Advection operator did not return conservation_error"
                )
            increments["diagnostics.conservation_error"] = conservation

        boundary_fluxes = result.get("boundary_fluxes")
        if boundary_fluxes:
            # References: LeVeque (2002), oriented finite-volume face fluxes.
            # Expression: j_out = (v dot n) c_upwind.
            # Parameters: v [m s-1], c [tracer_units], j_out [boundary_flux_units = tracer_units x m s-1].
            # Open route: the computation-layer packer retains x/y/z min/max
            # order without merging independent batch ledgers.
            boundary_cochain = pack_outward_boundary_flux_cochain(
                boundary_fluxes,
                squeeze_batch=bool(squeeze_dims),
            )
            increments[self.boundary_flux_key] = UpdateItem(
                value=boundary_cochain,
                mode=UpdateMode.REPLACE,
                unit=self.boundary_flux_units,
                basis="per_second",
                per_timestep=False,
                source_functor=self.name,
            )

        increments[self.tracer_key] = UpdateItem(
            value=delta_concentration,
            mode=UpdateMode.ACCUMULATE,
            unit=self.tracer_units,
            basis="per_step",
            per_timestep=True,
            source_functor=self.name,
        )

        return increments

    def _compute_iphs_advection_update(
        self,
        concentration: Tensor,
        concentration_op: Tensor,
        velocity: Union[Tensor, Tuple[Tensor, ...]],
        dt: float,
        boundary_state: Optional[Tensor],
    ) -> Dict[str, Any]:
        """Compute the canonical IPHS/DEC linear advection update.

        References:
          - Courant et al. (1928), CFL stability.
          - LeVeque (2007), finite-volume advection.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          c_dot = -D_h q_h + s_h, q_h = i_v c_h.
        Parameter meanings:
          concentration: c_h state cochain.
          velocity: v_h advective velocity [m s-1].
          dt: time step [s].
        Open/closed system route:
          Periodic transport is closed. Nonperiodic face exchange must be
          retained by the declared advective boundary port.
        """

        dt_effective = float(dt)

        boundary_method = self.boundary_method
        advection_operator = self._advection_operator
        if advection_operator is None:
            raise RuntimeError("Advection operator not initialized.")
        result = advection_operator.apply(
            concentration=concentration_op,
            velocity=velocity,
            grid_spacing=self._resolve_operator_spacing(concentration.shape),
            boundary_method=boundary_method,
            boundary_state=boundary_state,
            time_step=dt_effective,
            check_cfl=True,
        )
        return {
            "dt_effective": dt_effective,
            "operator_result": result,
        }

    # ------------------------------------------------------------------
    # Boundary condition helpers
    # ------------------------------------------------------------------
    def boundary_conditions(
        self, boundary: str, field: Tensor, normal_vector: Tensor
    ) -> Tensor:
        """Return the outward normal advective flux at one domain face.

        References:
          - LeVeque (2007), conservative finite-volume boundary fluxes.
          - Strikwerda (2004), normal flux boundary conditions.
        Mathematical expression:
          F_b = (v_b dot n_b) c_b.
        Parameter meanings:
          v_b: boundary velocity [m s-1]; n_b: outward unit normal [1];
          c_b: boundary tracer state [state_unit].
        """

        if self.velocity_field is None:
            raise RuntimeError("Velocity field not set")
        if not isinstance(boundary, str) or not boundary.strip():
            raise ValueError("boundary must be a non-empty face name")
        if not isinstance(field, Tensor) or not isinstance(normal_vector, Tensor):
            raise TypeError("field and normal_vector must be torch.Tensor values")

        field_boundary = field.to(self.device)
        normal = normal_vector.to(device=self.device, dtype=field_boundary.dtype)
        if not bool(torch.all(torch.isfinite(field_boundary))):
            raise ValueError("Boundary field must contain only finite values")
        if not bool(torch.all(torch.isfinite(normal))):
            raise ValueError("Boundary normal must contain only finite values")
        if normal.dim() == 0:
            normal = normal.reshape(1)
        if normal.shape[-1] != self.spatial_dims:
            raise ValueError(
                "Boundary normal last dimension must equal spatial_dims"
            )

        normal_velocity = torch.zeros_like(field_boundary)
        for axis, component in enumerate(self.velocity_field):
            component_boundary = self._select_boundary_velocity_component(
                component=component,
                boundary=boundary,
                field_shape=field_boundary.shape,
            ).to(device=self.device, dtype=field_boundary.dtype)
            normal_component = normal[..., axis]
            try:
                component_boundary = torch.broadcast_to(
                    component_boundary,
                    field_boundary.shape,
                )
                normal_component = torch.broadcast_to(
                    normal_component,
                    field_boundary.shape,
                )
            except RuntimeError as exc:
                raise ValueError(
                    "Boundary velocity and normal must be broadcast-compatible "
                    "with field"
                ) from exc
            if not bool(torch.all(torch.isfinite(component_boundary))):
                raise ValueError("Boundary velocity must contain only finite values")
            normal_velocity = normal_velocity + component_boundary * normal_component

        return normal_velocity * field_boundary

    def _select_boundary_velocity_component(
        self,
        component: Tensor,
        boundary: str,
        field_shape: torch.Size,
    ) -> Tensor:
        """Select one canonical x/y/z face from a configured velocity field."""

        if component.shape == field_shape or component.numel() == 1:
            return component

        face_indices = {
            "x_min": (-1, 0),
            "x_max": (-1, -1),
            "y_min": (-2, 0),
            "y_max": (-2, -1),
            "z_min": (-3, 0),
            "z_max": (-3, -1),
        }
        face = boundary.strip().lower()
        face_spec = face_indices.get(face)
        if face_spec is not None:
            axis, index = face_spec
            if component.dim() < abs(axis):
                raise ValueError(
                    f"Velocity component has no axis for boundary '{face}'"
                )
            selected = component.select(axis, index)
            try:
                torch.broadcast_shapes(selected.shape, field_shape)
                return selected
            except RuntimeError:
                pass

        try:
            torch.broadcast_shapes(component.shape, field_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Velocity component is incompatible with boundary '{face}'"
            ) from exc
        return component

    # ------------------------------------------------------------------
    # Dependency declarations
    # ------------------------------------------------------------------

    def get_provides(self) -> List[str]:
        return list(self.metadata.output_variables)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_concentration_from_state(self, state: StateDict) -> Tensor:
        if self.tracer_key not in state:
            raise ValueError(
                f"State must contain '{self.tracer_key}' for advection"
            )
        concentration = state[self.tracer_key]
        if not isinstance(concentration, Tensor):
            raise TypeError(f"State '{self.tracer_key}' must be a torch.Tensor")
        if concentration.dim() < self.spatial_dims:
            raise ValueError(
                f"Concentration tensor must have at least {self.spatial_dims} spatial dimensions."
            )
        if not bool(torch.all(torch.isfinite(concentration))):
            raise ValueError(f"State '{self.tracer_key}' must contain only finite values")
        return concentration.to(self.device)

    def _prepare_velocity_for_operator(self, state: StateDict) -> Union[Tensor, Tuple[Tensor, ...]]:
        if "velocity_field" in state:
            components = self._normalize_velocity_input(state["velocity_field"])
            if len(components) != self.spatial_dims:
                raise VelocityFieldError(
                    f"Need {self.spatial_dims} velocity components, received {len(components)}"
                )
            resolved = tuple(comp.to(self.device) for comp in components)
            concentration = self._extract_concentration_from_state(state)
            for component in resolved:
                if not bool(torch.all(torch.isfinite(component))):
                    raise ValueError("velocity_field must contain only finite values")
                try:
                    torch.broadcast_shapes(component.shape, concentration.shape)
                except RuntimeError as exc:
                    raise VelocityFieldError(
                        "velocity_field components must be broadcast-compatible "
                        f"with '{self.tracer_key}'"
                    ) from exc
            return resolved

        if self.velocity_field is not None:
            resolved = tuple(component.to(self.device) for component in self.velocity_field)
            if any(not bool(torch.all(torch.isfinite(component))) for component in resolved):
                raise ValueError("velocity_field must contain only finite values")
            return resolved

        raise RuntimeError("Velocity field not set")

    def _prepare_operator_inputs(
        self,
        concentration: Tensor,
        velocity: Union[Tensor, Tuple[Tensor, ...]],
    ) -> Tuple[Tensor, Union[Tensor, Tuple[Tensor, ...]], int]:
        squeeze_dims = 0
        concentration_op = concentration
        missing_dims = max(0, self.spatial_dims + 1 - concentration_op.dim())
        for _ in range(missing_dims):
            concentration_op = concentration_op.unsqueeze(0)
            squeeze_dims += 1

        if isinstance(velocity, tuple):
            velocity_op = tuple(
                component.unsqueeze(0) if component.dim() < concentration_op.dim() else component
                for component in velocity
            )
        else:
            velocity_op = velocity.unsqueeze(0) if velocity.dim() < concentration_op.dim() else velocity
        return concentration_op, velocity_op, squeeze_dims

    def _resolve_operator_spacing(self, concentration_shape: torch.Size) -> Union[float, Tuple[float, ...]]:
        if self.spatial_dims == 1:
            return float(self.grid_spacing[0])
        spacing = list(self.grid_spacing[: self.spatial_dims])
        return tuple(float(value) for value in spacing)

def create_uniform_advection(
    spatial_dims: int,
    velocity_magnitude: float,
    velocity_direction: Optional[Sequence[float]] = None,
    grid_spacing: Optional[Sequence[float]] = None,
    grid_shape: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> LinearAdvection:
    """Factory for uniform advection functor with constant flow."""

    if velocity_direction is None:
        velocity_direction = [1.0] + [0.0] * (spatial_dims - 1)
    if len(velocity_direction) != spatial_dims:
        raise ValueError(f"Direction must have {spatial_dims} components")

    direction_tensor = torch.as_tensor(velocity_direction, dtype=torch.float32)
    norm = torch.linalg.vector_norm(direction_tensor)
    if bool(norm <= 0):
        raise ValueError("Velocity direction vector must be non-zero")

    unit_direction = (direction_tensor / norm).tolist()
    advection = LinearAdvection(
        spatial_dims=spatial_dims,
        grid_spacing=grid_spacing,
        device=device,
    )

    velocity_components = []
    if grid_shape is None:
        base_shape = torch.Size([1] * spatial_dims)
    else:
        if len(grid_shape) != spatial_dims:
            raise ValueError(f"grid_shape must have {spatial_dims} entries")
        base_shape = torch.Size(int(max(1, points)) for points in grid_shape)
    for axis_idx in range(spatial_dims):
        component_value = velocity_magnitude * unit_direction[axis_idx]
        velocity_components.append(torch.full(base_shape, component_value, device=advection.device))
    advection.set_velocity_field(velocity_components)
    return advection


__all__ = ["LinearAdvection", "create_uniform_advection"]
