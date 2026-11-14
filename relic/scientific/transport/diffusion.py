r"""Fickian diffusion transport module for RELIC.

The class implemented here provides the diffusion functor API that is
exercised by the L1-L3 transport regression tests. It wraps the unified
diffusion operator provided in :mod:`relic.computation.operators.functional`
while enforcing the architectural guidelines of the FunctorBase workflow.

Mathematical Expressions
------------------------
1. Fick's Second Law: ``\partial c / \partial t = \nabla \cdot (K \nabla c) + S``
2. Stability number: ``D = K \Delta t / \Delta x^2`` (one-dimensional form)

References
----------
- Crank, J. (1975). *The Mathematics of Diffusion.* Oxford University Press.
- Bird, R.B., Stewart, W.E., Lightfoot, E.N. (2007). *Transport Phenomena.* Wiley.
- LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary and Partial
  Differential Equations.* SIAM.
"""

from __future__ import annotations

import logging
import math
from numbers import Integral
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, cast

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
from ...utils.physical_constants import get_physical_constant
from ...computation.operators import OperatorFactory
from ...computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from ...computation.operators.iphs import IPHSStructureSpec, PortSpecIPHS
from relic.computation.operators.functional import (
    DIFFUSIVITY_LAYOUTS,
    _compute_diffusion_rate_and_boundary_flux_1d,
    _normalize_directional_diffusivity,
    harmonic_mean,
    pack_outward_boundary_flux_cochain,
)

logger = logging.getLogger(__name__)


class DiffusivityError(ValueError):
    """Raised when the supplied diffusivity is inconsistent with the grid."""


SUPPORTED_DIFFUSION_BOUNDARIES = {
    "zero",
    "neumann",
    "periodic",
    "open",
    "dirichlet",
    "extrapolate",
}


class FickianDiffusion(TransportMechanismFunctor):
    """Pure diffusion transport functor with parameter-manager integration.

    Parameters
    ----------
    spatial_dims : int, optional
        Number of spatial dimensions (1, 2, or 3). Units: dimensionless.
    diffusivity : Optional[Union[float, Tensor]]
        Diffusion coefficient ``K`` (``m^2 s^-1``). If omitted the default
        value registered under ``transport.diffusion.diffusivity`` is used.
        A number or a 0-d tensor is a scalar ``D`` and is also stored as the
        base ``TransportMechanism`` scalar; a tensor with ``ndim >= 1`` is a
        field and leaves the base scalar ``None``.
    diffusivity_layout : Optional[str]
        ``"scalar_field"`` or ``"directional"``; required for a tensor
        diffusivity with ``ndim >= 1`` (it is never inferred from the shape).
        A scalar ``D`` has the ``"scalar_field"`` layout.
    grid_spacing : Optional[Union[float, Sequence[float]]]
        Grid spacing per spatial axis (``m``).
    non_negative : Optional[bool]
        Whether to enforce non-negative concentrations after the update.
    track_conservation : Optional[bool]
        Enable conservation diagnostics using
        ``transport.numerical.conservation_tolerance``.
    device : Optional[torch.device]
        Target computation device.
    diffusivity_parameter_key : str, optional
        Fully qualified ParameterManager key used when ``diffusivity`` is
        omitted. Runtime metadata and IPHS Hodge evidence use the same key.
    grid_spacing_parameter_key : str, optional
        Fully qualified ParameterManager key used when ``grid_spacing`` is
        omitted. Runtime metadata and IPHS metric evidence use the same key.
    """

    TRANSPORT_TYPE = "diffusion"

    ALLOWED_TIME_INTEGRATION = {"explicit", "semi_implicit"}
    CLOSED_BOUNDARY_METHODS = frozenset({"zero", "neumann", "open", "periodic"})
    CANONICAL_DRIFT_SOURCE = (
        "relic.computation.operators.iphs.IPHSStructureSpec.canonical_drift"
    )
    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    DEFAULT_DIFFUSIVITY_PARAMETER_KEY = "transport.diffusion.diffusivity"
    DEFAULT_GRID_SPACING_PARAMETER_KEY = "transport.diffusion.grid_spacing"

    def __init__(
        self,
        spatial_dims: int = 1,
        diffusivity: Optional[Union[float, Tensor]] = None,
        grid_spacing: Optional[Union[float, Sequence[float]]] = None,
        boundary_method: Optional[str] = None,
        non_negative: Optional[bool] = None,
        track_conservation: Optional[bool] = None,
        device: Optional[torch.device] = None,
        name: str = "fickian_diffusion",
        config: Optional[Dict[str, Union[str, float, int, bool]]] = None,
        time_integration: str = "explicit",
        diffusivity_parameter_key: str = DEFAULT_DIFFUSIVITY_PARAMETER_KEY,
        grid_spacing_parameter_key: str = DEFAULT_GRID_SPACING_PARAMETER_KEY,
        strict_runtime_point: bool = False,
        diffusivity_layout: Optional[str] = None,
    ) -> None:
        self._validate_spatial_dims(spatial_dims)
        if not isinstance(strict_runtime_point, bool):
            raise ValueError("strict_runtime_point must be a bool selection field.")

        if time_integration not in self.ALLOWED_TIME_INTEGRATION:
            raise ValueError(
                f"Time integration must be one of {sorted(self.ALLOWED_TIME_INTEGRATION)}, "
                f"got '{time_integration}'."
            )

        selected_diffusivity_key = self._validate_diffusivity_parameter_key(
            diffusivity_parameter_key
        )
        selected_grid_spacing_key = self._validate_grid_spacing_parameter_key(
            grid_spacing_parameter_key
        )
        metadata = self._create_metadata(
            name,
            selected_diffusivity_key,
            selected_grid_spacing_key,
        )
        transport_kwargs = {
            "device": device if device is not None else torch.device("cpu"),
        }
        super().__init__(name=name, config=config or {}, metadata=metadata, **transport_kwargs)

        self.device = device if device is not None else torch.device("cpu")
        self.spatial_dims = spatial_dims
        self.diffusivity_parameter_key = selected_diffusivity_key
        self.grid_spacing_parameter_key = selected_grid_spacing_key
        self.tracer_key = "tracer_concentration"
        self.source_key = "source_term"
        self.boundary_state_key = f"{self.tracer_key}_boundary_state"
        self.boundary_flux_key = f"{self.tracer_key}_diffusive_boundary_flux"
        self.optional_input_variables = {self.source_key, self.boundary_state_key}
        self.boundary_method = self._resolve_boundary_method(boundary_method)
        # Level 4 dPH runtime point: a GraphIR selection field, not a parameter.
        # It never changes the production law; it audits each emitted step and
        # rejects a nonzero source_term, which the source-free audit cannot cover.
        if strict_runtime_point and (
            spatial_dims != 1 or self.boundary_method not in self.CLOSED_BOUNDARY_METHODS
        ):
            raise ValueError(
                "strict_runtime_point requires spatial_dims=1 and a closed boundary "
                f"method {sorted(self.CLOSED_BOUNDARY_METHODS)}; got spatial_dims="
                f"{spatial_dims}, boundary_method={self.boundary_method!r}."
            )
        self.strict_runtime_point = strict_runtime_point
        self._last_strict_iphs_audit: Optional[Dict[str, Any]] = None

        self.operator_factory = OperatorFactory()
        self._diffusion_operator = None

        self.grid_spacing = self._initialize_grid_spacing(grid_spacing)
        self._initialize_diffusivity(diffusivity, diffusivity_layout)

        numerical_params = self._load_numerical_parameters()
        if non_negative is None:
            non_negative = bool(numerical_params["non_negative"])
        self.non_negative = bool(non_negative)

        if track_conservation is None:
            track_conservation = bool(numerical_params["track_conservation"])
        self.track_conservation = bool(track_conservation)
        self.time_integration = time_integration
        self._initialize_diffusion_operator(name)

    # ------------------------------------------------------------------
    # Metadata and parameter helpers
    # ------------------------------------------------------------------
    def _resolve_boundary_method(self, explicit: Optional[str]) -> str:
        """Resolve the single finite-volume diffusion boundary closure."""

        selected: Any = explicit
        if selected is None:
            selected = self.get_parameter_value(
                "boundary_method",
                namespace="transport.diffusion",
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
        if normalized not in SUPPORTED_DIFFUSION_BOUNDARIES:
            raise ValueError(
                "FickianDiffusion boundary_method must be one of "
                f"{sorted(SUPPORTED_DIFFUSION_BOUNDARIES)}"
            )
        return normalized

    def _create_metadata(
        self,
        name: str,
        diffusivity_parameter_key: str,
        grid_spacing_parameter_key: str,
    ) -> FunctorMetadata:
        tracer_key = "tracer_concentration"
        source_key = "source_term"
        boundary_state_key = f"{tracer_key}_boundary_state"
        boundary_flux_key = f"{tracer_key}_diffusive_boundary_flux"
        parameters = {
            diffusivity_parameter_key: "Selected isotropic diffusivity [m2 s-1]",
            grid_spacing_parameter_key: "Selected uniform grid spacing [m]",
            "transport.diffusion.temperature_exponent": "Gas diffusivity temperature exponent [dimensionless]",
            "transport.numerical.conservation_tolerance": "Relative conservation tolerance [dimensionless]",
            "transport.numerical.cfl_diffusion_max": "Explicit diffusion stability limit [dimensionless]",
            "transport.numerical.enforce_non_negative": "Non-negative state-domain guard [bool]",
            "transport.numerical.track_conservation": "Finite-volume conservation audit flag [bool]",
            "transport.numerical.default_boundary_type": "Default boundary method [categorical]",
        }

        return FunctorMetadata(
            name=name,
            version="1.0.0",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            description="Fickian diffusion transport mechanism with isotropic/anisotropic support.",
            mathematical_expression=(
                "c_dot = -D_h q_h + G_h u_h; q_h = -star_K d_h c_h; "
                "G_h u_h is source_term plus routed boundary/open flux"
            ),
            references=[
                "Crank (1975) The Mathematics of Diffusion",
                "Bird et al. (2007) Transport Phenomena",
                "LeVeque (2007) Finite Difference Methods for ODE/PDE",
            ],
            conservation_laws=["mass"],
            parameters=parameters,
            input_variables=[tracer_key, source_key, boundary_state_key],
            output_variables=[
                tracer_key,
                boundary_flux_key,
                "diagnostics.diffusion_stability",
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
        parameter_specs: Dict[str, Tuple[str, Optional[str], type]] = {
            "non_negative": ("enforce_non_negative", None, bool),
            "track_conservation": ("track_conservation", None, bool),
        }

        params: Dict[str, Union[float, bool]] = {}
        for attr, (name, units, expected_type) in parameter_specs.items():
            value = self.get_parameter_value(
                name,
                namespace="transport.numerical",
                units=units,
                expected_type=expected_type,
                source_order=("manager",),
                required=True,
            )
            if expected_type is bool:
                params[attr] = bool(value)
            else:
                params[attr] = float(value)

        return params

    def _structural_n_dofs(
        self,
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> int:
        """Resolve local DEC degrees of freedom for structural audits."""

        if n_dofs is not None:
            declared_value = n_dofs
        elif hasattr(complex_bundle, "n_dofs"):
            declared_value = getattr(complex_bundle, "n_dofs")
        else:
            declared_value = 1
        if isinstance(declared_value, bool) or not isinstance(declared_value, Integral):
            raise TypeError("Structural degrees of freedom must be an integer.")
        value = int(declared_value)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for Fickian diffusion.

        References:
          - Crank (1975), The Mathematics of Diffusion.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          c_dot = -D_h q_h + G_h u_h, q_h = -star_K d_h c_h.
        Parameter meanings:
          n_dofs is a structural audit size; source_term and boundary flux
          form the open input contribution G_h u_h.
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="transport_fickian_diffusion",
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
                complex_id="transport_fickian_diffusion",
                degree=1,
                primality="primal",
                n_dofs=n_dofs,
                units="state_unit m-1",
                extensivity="intensive",
                physical_domain="transport",
                semantic_name="tracer_gradient",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="transport_fickian_diffusion",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="state_unit m s-1",
                extensivity="integral",
                physical_domain="transport",
                semantic_name="diffusive_tracer_flux",
                orientation_convention="outward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            self.diffusivity_parameter_key,
            self.grid_spacing_parameter_key,
            "transport.diffusion.temperature_exponent",
            "transport.numerical.conservation_tolerance",
            "transport.numerical.enforce_non_negative",
            "transport.numerical.track_conservation",
            "transport.numerical.cfl_diffusion_max",
            "transport.numerical.default_boundary_type",
            "transport.numerical.implicit_midpoint_max_iter",
            "transport.numerical.implicit_midpoint_tolerance",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
        diffusivity: Optional[Union[float, Tensor]] = None,
    ) -> HodgeStarSpec:
        """Build the diffusion Hodge contract.

        References:
          - Crank (1975), diffusion flux law.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          q_h = -star_K d_h c_h with star_K = diag(K_i) and K_i > 0.
        Parameter meanings:
          diffusivity_scalar or diffusivity_tensor supplies K [m2 s-1].
          grid_spacing supplies the metric support [m].
        Units: K [m2 s-1]; this local material map is not a global FV Laplacian.
        Domain/constraints: scalar K or an explicit coefficient for every DOF;
          no directional or cell coefficient may be silently discarded.
        Sign convention: the minus sign in q_h = -star_K d_h c_h is external
          to the positive material matrix returned here.
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        coefficient = self.get_diffusivity() if diffusivity is None else diffusivity
        if not isinstance(coefficient, Tensor):
            values = torch.full(
                (n_dofs,),
                float(coefficient),
                dtype=torch.get_default_dtype(),
                device=self.device,
            )
        else:
            if not coefficient.is_floating_point():
                raise DiffusivityError("Diffusivity must use a real floating-point dtype.")
            values = coefficient.to(self.device).reshape(-1)
            if values.numel() == 1 and n_dofs > 1:
                values = values.expand(n_dofs)
            elif values.numel() != n_dofs:
                raise DiffusivityError(
                    f"Cannot map {values.numel()} diffusivity coefficients to {n_dofs} DOFs without an explicit mapping."
                )
        if not bool(torch.all(torch.isfinite(values))) or bool(torch.any(values <= 0.0)):
            raise DiffusivityError("Diffusivity must be finite and positive for Hodge assembly.")

        domain = CochainSpaceSpec(
            complex_id="transport_fickian_diffusion",
            degree=1,
            primality="primal",
            n_dofs=n_dofs,
            units="state_unit m-1",
            extensivity="intensive",
            physical_domain="transport",
            semantic_name="tracer_gradient",
            orientation_convention="outward_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="transport_fickian_diffusion",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="state_unit m s-1",
            extensivity="integral",
            physical_domain="transport",
            semantic_name="diffusive_tracer_flux",
            orientation_convention="outward_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=(self.diffusivity_parameter_key,),
            metric_keys=(self.grid_spacing_parameter_key,),
            material_law="fickian_diffusion",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build diffusion Hodge terms for IPHS assembly."""

        key = self.diffusivity_parameter_key
        if resolved_parameters and key in resolved_parameters:
            coefficient = resolved_parameters[key]
            if coefficient is None:
                raise TypeError("Resolved diffusivity must be a scalar or Tensor")
        else:
            coefficient = None
        return (
            self.build_hodge_contract(
                n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
                strict=True,
                diffusivity=coefficient,
            ),
        )

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return storage and effort terms for the level 3 structural audit."""

        if not self._has_tracer_state(state):
            return {
                "storage_functional": "H_h = 0.5 * c_h.T c_h",
                "effort": "e_h = delta H_h / delta c_h = c_h",
            }
        concentration = self._extract_concentration(dict(state))
        concentration, _ = self._ensure_batch_dimension(concentration)
        return {
            "storage_functional": self._quadratic_storage(concentration),
            "effort": concentration,
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return non-negative dissipation for the level 3 structural audit."""

        if not self._has_tracer_state(state):
            return {
                "dissipation_rate": "sigma_h = X_h.T star_K X_h",
                "thermodynamic_force": "X_h = d_h c_h",
            }
        concentration = self._extract_concentration(dict(state))
        concentration, _ = self._ensure_batch_dimension(concentration)
        return {
            "dissipation_rate": self._diffusion_dissipation_rate(
                concentration, boundary_state=state.get(self.boundary_state_key),
            ),
            "thermodynamic_force": "X_h = d_h c_h",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for pure Fickian diffusion."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive diffusion mobility contribution.

        References:
          - Crank (1975), diffusion flux law.
          - van der Schaft and Maschke (1995), dissipative port-Hamiltonian form.
        Mathematical expression:
          M_h = diag(K_i), K_i > 0, sigma = X_h.T M_h X_h >= 0.
        Parameter meanings:
          K_i is the selected diffusivity ParameterManager value [m2 s-1].
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"{self.tracer_key}[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the boundary diffusive flux port.

        References:
          - Crank (1975), diffusive boundary fluxes.
          - van der Schaft and Maschke (1995), port variables and power pairing.
        Mathematical expression:
          P_port = e_h * f_h, with e_h = c_boundary and f_h = q_boundary.
        Parameter meanings:
          The StateManager supplies the explicit boundary state when required.
          Positive flow denotes outward matter export from the modeled domain.
        """

        effort_space = CochainSpaceSpec(
            complex_id="transport_fickian_diffusion",
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
            complex_id="transport_fickian_diffusion",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="state_unit m s-1",
            extensivity="integral",
            physical_domain="transport",
            semantic_name="boundary_diffusive_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="diffusive_boundary_flux",
                effort_var=f"{self.tracer_key}_boundary_state",
                flow_var=self.boundary_flux_key,
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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _scalar_from_value(value: Union[Tensor, float, int]) -> float:
        """Convert numeric values to Python float without scalar extraction helpers that sync GPU."""
        if isinstance(value, Tensor):
            detached = value.detach().reshape(-1)
            if detached.numel() == 0:
                raise ValueError("Cannot convert empty tensor to scalar.")
            return detached[:1].to(dtype=torch.float64).tolist()[0]
        return float(value)

    @staticmethod
    def _validate_spatial_dims(spatial_dims: int) -> None:
        if spatial_dims not in (1, 2, 3):
            raise ValueError("Spatial dimensions must be 1, 2, or 3")

    @staticmethod
    def _validate_diffusivity_parameter_key(parameter_key: str) -> str:
        """Validate the one fully qualified diffusivity parameter binding."""

        if not isinstance(parameter_key, str):
            raise TypeError("diffusivity_parameter_key must be a string")
        normalized = parameter_key.strip()
        if (
            "." not in normalized
            or normalized.startswith(".")
            or normalized.endswith(".")
            or ".." in normalized
        ):
            raise ValueError(
                "diffusivity_parameter_key must be a fully qualified parameter key"
            )
        return normalized

    @staticmethod
    def _validate_grid_spacing_parameter_key(parameter_key: str) -> str:
        """Validate the one fully qualified grid-spacing parameter binding."""

        if not isinstance(parameter_key, str):
            raise TypeError("grid_spacing_parameter_key must be a string")
        normalized = parameter_key.strip()
        if (
            "." not in normalized
            or normalized.startswith(".")
            or normalized.endswith(".")
            or ".." in normalized
        ):
            raise ValueError(
                "grid_spacing_parameter_key must be a fully qualified "
                "parameter key"
            )
        return normalized

    # ------------------------------------------------------------------
    # Grid and diffusivity setup
    # ------------------------------------------------------------------
    def _initialize_grid_spacing(
        self, grid_spacing: Optional[Union[float, Sequence[float]]]
    ) -> List[float]:
        if grid_spacing is None:
            default_spacing = self.get_parameter_value(
                self.grid_spacing_parameter_key,
                required=True,
            )
            spacing = [float(default_spacing)] * self.spatial_dims
        elif isinstance(grid_spacing, (int, float)):
            spacing = [float(grid_spacing)] * self.spatial_dims
        else:
            spacing = [float(value) for value in grid_spacing]
            if len(spacing) != self.spatial_dims:
                raise ValueError(
                    f"Grid spacing must contain {self.spatial_dims} values, received {len(spacing)}"
                )
        return spacing

    def _initialize_diffusivity(
        self,
        diffusivity: Optional[Union[float, Tensor]],
        layout: Optional[str],
    ) -> None:
        parameter_bound = diffusivity is None
        if diffusivity is None:
            diffusivity = self.get_parameter_value(
                self.diffusivity_parameter_key,
                required=True,
            )
        if diffusivity is None:
            raise DiffusivityError("Required diffusivity parameter did not resolve a value")

        self.set_diffusivity(diffusivity, layout=layout)
        self._diffusivity_parameter_bound = parameter_bound

    def _initialize_diffusion_operator(self, name: str) -> None:
        self._diffusion_operator = self.operator_factory.get_operator(
            "transport.diffusion",
            name=f"{name}_operator",
            config={
                "spatial_dims": self.spatial_dims,
                "boundary_method": "zero",
                "check_stability": True,
                "enforce_non_negative": False,
            },
            device=self.device,
            dtype=torch.float32,
        )

    def set_diffusivity(
        self,
        diffusivity: Union[float, Tensor],
        layout: Optional[str] = None,
    ) -> None:
        """Set the diffusivity K with its declared layout and the base scalar D.

        References:
          - Crank (1975), The Mathematics of Diffusion, Ch. 1 (Fick's first law
            with a constant scalar D).
          - Patankar (1980), Numerical Heat Transfer and Fluid Flow (per-axis
            finite-volume coefficients).
        Mathematical expression:
          J_k = -K_k dc/dx_k, with K_k = D for a scalar and K_k the declared
          per-axis reading of a tensor K.
        Parameter meanings:
          diffusivity is D or K; layout is "scalar_field" (one coefficient or
          one cell field shared by every axis) or "directional" (storage-ordered
          (Kz, Ky, Kx) values or fields).
        Units:
          D and K [m2 s-1].
        Domain/constraints:
          finite and > 0. A number or a 0-d tensor is a scalar D: its layout is
          "scalar_field" and it is stored as the base TransportMechanism scalar
          through set_scalar_diffusivity (a tensor, including a learnable
          Parameter, as the same object, never copied to a float). A tensor with
          ndim >= 1 is a field: its layout must be declared (it is never
          inferred from the shape) and the base scalar is None.
        Sign convention:
          K > 0; the flux direction comes from -grad(c).
        """

        if layout is not None and layout not in DIFFUSIVITY_LAYOUTS:
            raise ValueError(
                f"Unknown diffusivity_layout '{layout}'; expected one of {DIFFUSIVITY_LAYOUTS}"
            )
        if isinstance(diffusivity, Tensor):
            if not diffusivity.is_floating_point():
                raise DiffusivityError("Diffusivity must use a real floating-point dtype")
            if not bool(torch.all(torch.isfinite(diffusivity))) or bool(torch.any(diffusivity <= 0)):
                raise DiffusivityError("Diffusivity must be positive and finite")
            tensor = diffusivity.to(self.device)
            scalar = self._scalar_from_value(tensor.detach().mean())
        else:
            scalar = float(diffusivity)
            if not math.isfinite(scalar) or scalar <= 0:
                raise DiffusivityError("Diffusivity must be positive and finite")
            tensor = None
        is_scalar = tensor is None or int(tensor.ndim) == 0
        if is_scalar:
            if layout not in (None, "scalar_field"):
                raise DiffusivityError(
                    f"A scalar diffusivity has the 'scalar_field' layout, not '{layout}'"
                )
            resolved_layout = "scalar_field"
        else:
            if layout is None:
                raise DiffusivityError(
                    f"A tensor diffusivity of shape {tuple(tensor.shape)} needs "
                    "diffusivity_layout='scalar_field' or 'directional'; the layout "
                    "is not inferred from the shape"
                )
            resolved_layout = layout
        # Replace an optional Module parameter through its canonical lifecycle.
        if "diffusivity_tensor" in self._parameters:
            delattr(self, "diffusivity_tensor")
        self.diffusivity_tensor = tensor
        self.diffusivity_scalar = scalar
        self.diffusivity_layout = resolved_layout
        # A parameter-bound or explicit scalar diffusivity is isotropic; only a
        # directional or cell field is anisotropic.
        self.isotropic = is_scalar
        # The base scalar D of TransportMechanism: set through its one validated
        # setter for a scalar (the same object for a tensor), None for a field.
        if "diffusivity" in self._parameters:
            delattr(self, "diffusivity")
        if is_scalar:
            self.set_scalar_diffusivity(tensor if tensor is not None else scalar)
        else:
            self.diffusivity = None
        self._diffusivity_parameter_bound = False

    def get_diffusivity(
        self,
        temperature: Optional[Tensor] = None,
        pressure: Optional[Tensor] = None,
    ) -> Tensor:
        """Resolve the active coefficient and optional temperature/pressure scaling.

        Manager-bound coefficients are read through the existing parameter API
        on each evaluation; explicit constructor/setter values stay explicit.
        No scalar conversion detaches a learnable coefficient from its owner.
        """

        if self._diffusivity_parameter_bound:
            base = self.get_parameter_value(self.diffusivity_parameter_key, required=True)
            if not isinstance(base, Tensor):
                raise DiffusivityError("Bound diffusivity must resolve to a floating-point Tensor")
            base = base.to(self.device)
        elif self.diffusivity_tensor is not None:
            base = self.diffusivity_tensor.to(self.device)
        else:
            base = torch.tensor(self.diffusivity_scalar, device=self.device)
        if not base.is_floating_point() or not bool(torch.all(torch.isfinite(base))) or bool(torch.any(base <= 0)):
            raise DiffusivityError("Diffusivity must be positive, finite, and floating-point")

        if temperature is None and pressure is None:
            return base

        # References: Fuller et al. (1966); Bird et al. (2007).
        # Expression: D = D0 * (T/T_ref)^n * (P_ref/P)
        # Parameters: D0 base diffusivity (m2 s-1), T/T_ref (K), n exponent (dimensionless),
        # P/P_ref (Pa).
        scale = torch.ones((), device=self.device, dtype=base.dtype)
        if temperature is not None:
            temp_tensor = torch.as_tensor(temperature, device=self.device, dtype=base.dtype)
            t_ref = get_physical_constant("reference_temperature")
            exponent = self.get_parameter_value(
                "temperature_exponent",
                namespace="transport.diffusion", required=True)
            t_ref_tensor = torch.as_tensor(t_ref, device=self.device, dtype=base.dtype)
            exponent_tensor = torch.as_tensor(exponent, device=self.device, dtype=base.dtype)
            scale = scale * (temp_tensor / t_ref_tensor) ** exponent_tensor

        if pressure is not None:
            pressure_tensor = torch.as_tensor(pressure, device=self.device, dtype=base.dtype)
            p_ref = get_physical_constant("reference_pressure")
            p_ref_tensor = torch.as_tensor(p_ref, device=self.device, dtype=base.dtype)
            scale = scale * (p_ref_tensor / pressure_tensor)

        return base * scale

    # ------------------------------------------------------------------
    # Core functor interfaces
    # ------------------------------------------------------------------
    def last_strict_iphs_audit(self) -> Optional[Dict[str, Any]]:
        """Return the Level 4 runtime-point audit of the last emitted step."""

        audit = self._last_strict_iphs_audit
        return None if audit is None else dict(audit)

    def _closed_dph_runtime_point(
        self,
        concentration: Tensor,
        diffusivity: Tensor,
    ) -> Dict[str, Any]:
        """Verify the closed finite-volume diffusion law as a Level 4 dPH runtime point.

        References:
          - van der Schaft and Jeltsema (2014), Port-Hamiltonian Systems Theory.
          - Patankar (1980), Numerical Heat Transfer and Fluid Flow.
          - Crank (1975), The Mathematics of Diffusion.
        Mathematical expression:
          H(c) = 0.5 c^T c, e_H = dH/dc = c;
          d in R^{faces x cells} cell-to-face difference, K_f = harmonic(K_i, K_j);
          R = d^T diag(K_f) d / dx^2, J = 0, M = 0, G = 0;
          x_dot = -R e_H = -R c, which equals the closed FV rate (right - left)/dx;
          dH/dt = -e_H^T R e_H = -sigma <= 0, sigma = sum_f K_f (dc_f / dx)^2.
        Parameter meanings:
          c cell concentrations, K cell diffusivities, dx grid spacing, faces
          are interior faces plus the periodic seam; runtime_point_tolerance is
          the shared SSOT audit tolerance.
        Units:
          c kg m-3, K m2 s-1, dx m, R s-1, x_dot kg m-3 s-1, H kg2 m-6,
          sigma and dH/dt kg2 m-6 s-1 (quadratic-norm storage, not joules).
        Domain/constraints:
          one spatial dimension, one unbatched rank-1 state, closed boundary
          (zero, neumann, open, periodic); finite positive diffusivities; no
          nonzero source_term (forward rejects it before integration, so the
          executed step equals the audited source-free drift).
        Sign convention:
          dissipation power positive, closed-port energy rate negative; the
          audit reproduces the production FV rate and must match it.
        """

        if concentration.dim() != 1:
            raise ValueError(
                "strict_runtime_point audits a single unbatched one-dimensional state; "
                f"got shape {tuple(concentration.shape)}."
            )
        n_cells = int(concentration.numel())
        if n_cells < 2:
            raise ValueError("strict_runtime_point requires at least two cells.")
        cell_diffusivity = _normalize_directional_diffusivity(
            diffusivity.to(device=concentration.device, dtype=concentration.dtype),
            concentration,
            1,
            self.diffusivity_layout,
        )[0].reshape(-1)
        if cell_diffusivity.numel() != n_cells:
            raise ValueError("strict_runtime_point requires one diffusivity per cell.")
        if not bool(torch.all(torch.isfinite(cell_diffusivity))) or bool(torch.any(cell_diffusivity <= 0.0)):
            raise ValueError("strict_runtime_point requires finite positive diffusivities.")
        dx = float(self.grid_spacing[0])
        tolerance = float(
            self.get_parameter_value(
                "runtime_point_tolerance",
                namespace="validation.iphs",
                source_order=("manager",),
                required=True,
            )
        )

        left_cells = list(range(n_cells - 1))
        right_cells = list(range(1, n_cells))
        if self.boundary_method == "periodic":
            left_cells.append(n_cells - 1)
            right_cells.append(0)
        n_faces = len(left_cells)
        difference = torch.zeros(
            (n_faces, n_cells), dtype=concentration.dtype, device=concentration.device
        )
        face_index = torch.arange(n_faces, device=concentration.device)
        difference[face_index, torch.tensor(left_cells, device=concentration.device)] = -1.0
        difference[face_index, torch.tensor(right_cells, device=concentration.device)] = 1.0
        face_diffusivity = harmonic_mean(
            cell_diffusivity[left_cells], cell_diffusivity[right_cells]
        )
        resistance = (
            difference.transpose(0, 1) @ (face_diffusivity.unsqueeze(1) * difference)
        ) / (dx * dx)

        state = concentration.detach().clone().requires_grad_(True)
        storage = 0.5 * torch.sum(state * state)
        spec = IPHSStructureSpec(
            name=f"{self.name}.closed_dph_runtime_point",
            state_vars=tuple(f"{self.tracer_key}[{index}]" for index in range(n_cells)),
            energy_functional="H = 0.5 c.c",
            entropy_functional="not_applicable_dissipative_ph",
            J=torch.zeros_like(resistance),
            M=torch.zeros_like(resistance),
            R=resistance,
            G=None,
            stoichiometric_N=None,
            ports=(),
            formulation="dissipative_ph",
            poisson_operator="zero_field",
            reaction_representation="independent",
            skew_tol=tolerance,
            psd_tol=tolerance,
            degeneracy_tol=tolerance,
            contribution_count=1,
            contribution_sources=(self.CANONICAL_DRIFT_SOURCE,),
        )
        drift = spec.canonical_drift(state=state, energy_value=storage)
        fv_rate, _ = _compute_diffusion_rate_and_boundary_flux_1d(
            concentration.detach(),
            cell_diffusivity.detach(),
            dx,
            self.boundary_method,
        )
        scale = torch.linalg.vector_norm(fv_rate.detach())
        residual = torch.linalg.vector_norm(drift.detach() - fv_rate.detach())
        machine_epsilon = torch.finfo(concentration.dtype).eps
        relative_tolerance = max(tolerance, (8 * n_cells + 16) * machine_epsilon)
        if bool(residual > relative_tolerance * torch.clamp(scale, min=1.0)):
            raise ValueError(
                "strict_runtime_point: canonical dPH drift -R e_H does not reproduce "
                f"the production finite-volume rate; residual={float(residual):.6g}."
            )
        audit = spec.validate(required_level=4, state=state, energy_value=storage)
        audit["fv_rate_residual"] = float(residual)
        audit["fv_rate_scale"] = float(scale)
        audit["state_size"] = n_cells
        audit["boundary_method"] = self.boundary_method
        return audit

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """Advance one conservative finite-volume diffusion step.

        References:
          - Crank (1975), The Mathematics of Diffusion.
          - Patankar (1980), Numerical Heat Transfer and Fluid Flow.
        Mathematical expression:
          dc/dt = div(K grad(c)) + S and Delta c = c_next - c.
        Parameter meanings:
          c is tracer concentration [kg m-3], K is diffusivity [m2 s-1],
          S is the optional source [kg m-3 s-1], and dt is time [s].
        Open/closed system route:
          Source input and outward-positive Dirichlet or extrapolated face
          fluxes are retained by the functional mass ledger and boundary port.
        """

        # References: Crank (1975); Patankar (1980).
        # Mathematical expression: dc/dt = div(K grad(c)) + S.
        # Parameter meanings: c [kg m-3], K [m2 s-1], S [kg m-3 s-1].
        if kwargs:
            raise TypeError(f"Unsupported diffusion forward arguments: {sorted(kwargs)}")
        _ = time
        self._last_strict_iphs_audit = None
        if not math.isfinite(float(dt)) or float(dt) <= 0.0:
            raise ValueError("Time step must be finite and positive")

        concentration = self._extract_concentration(state)
        if self.non_negative and bool(torch.any(concentration < 0.0)):
            raise ValueError(
                "FickianDiffusion requires a non-negative concentration state"
            )
        concentration_op, squeeze_dims = self._ensure_batch_dimension(concentration)
        source_term, boundary_state = self._validated_source_and_boundary(state)

        operator = self._diffusion_operator
        if operator is None:
            raise RuntimeError("Diffusion operator not initialized")
        diffusivity = self.get_diffusivity()
        integration_method = (
            "euler" if self.time_integration == "explicit" else "implicit_midpoint"
        )
        raw_result = operator.apply(
            concentration=concentration_op,
            diffusivity=diffusivity,
            diffusivity_layout=self.diffusivity_layout,
            grid_spacing=self._operator_spacing(),
            spatial_dims=self.spatial_dims,
            boundary_method=self.boundary_method,
            boundary_state=boundary_state,
            source_term=source_term,
            time_step=float(dt),
            check_stability=True,
            enforce_non_negative=self.non_negative,
            run_conservation_check=self.track_conservation,
            time_integration_method=integration_method,
        )
        if not isinstance(raw_result, Mapping):
            raise TypeError("Diffusion operator must return a result mapping")
        result = cast(Mapping[str, Any], raw_result)
        concentration_new = result.get("concentration")
        if concentration_new is None:
            raise RuntimeError("Diffusion operator did not return concentration")
        for _ in range(squeeze_dims):
            concentration_new = concentration_new.squeeze(0)
        concentration_new = concentration_new.to(
            device=concentration.device,
            dtype=concentration.dtype,
        )
        delta_concentration = concentration_new - concentration
        strict_audit = (
            self._closed_dph_runtime_point(concentration, diffusivity)
            if self.strict_runtime_point
            else None
        )

        stability_number = result.get("stability_number")
        if stability_number is None:
            raise RuntimeError("Diffusion operator did not return stability_number")
        increments: FluxDict = {
            "diagnostics.diffusion_stability": torch.max(stability_number),
        }
        if self.track_conservation:
            conservation_error = result.get("conservation_error")
            if conservation_error is None:
                raise RuntimeError("Diffusion operator did not return conservation_error")
            increments["diagnostics.conservation_error"] = conservation_error

        boundary_fluxes = result.get("boundary_fluxes")
        if boundary_fluxes:
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

        increments[self.tracer_key] = UpdateItem(
            value=delta_concentration,
            mode=UpdateMode.ACCUMULATE,
            unit="kg m-3",
            basis="per_step",
            per_timestep=True,
            source_functor=self.name,
        )
        if strict_audit is not None:
            self._last_strict_iphs_audit = strict_audit
        return increments

    def _validated_source_and_boundary(
        self, state: StateDict,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Return the optional source term and boundary state after their input checks.

        The source must be a finite tensor and is refused when nonzero at a strict
        runtime point; the boundary state is required exactly for Dirichlet closure
        and must be a finite (and, for non-negative tracers, non-negative) tensor.
        """

        source_term = state.get(self.source_key)
        if source_term is not None:
            if not isinstance(source_term, Tensor):
                raise TypeError(f"State '{self.source_key}' must be a torch.Tensor")
            if not bool(torch.all(torch.isfinite(source_term))):
                raise ValueError(f"State '{self.source_key}' must contain only finite values")
            # The Level 4 runtime point certifies the closed source-free drift
            # c_dot = -R_h e_H; a nonzero source would be integrated by the
            # production step outside that audited drift, so it is refused.
            if self.strict_runtime_point and bool(torch.any(source_term != 0.0)):
                raise ValueError(
                    "strict_runtime_point certifies the closed source-free drift "
                    f"c_dot = -R_h e_H; a nonzero '{self.source_key}' lies outside the "
                    "audited drift and is rejected. Use strict_runtime_point=False to "
                    "integrate a source term."
                )

        boundary_state = state.get(self.boundary_state_key)
        requires_boundary_state = self.boundary_method == "dirichlet"
        if requires_boundary_state and boundary_state is None:
            raise ValueError("Dirichlet diffusion requires explicit boundary_state")
        if boundary_state is not None and not requires_boundary_state:
            raise ValueError(
                "boundary_state is only supported for Dirichlet diffusion"
            )
        if boundary_state is not None:
            if not isinstance(boundary_state, Tensor):
                raise TypeError(
                    f"State '{self.boundary_state_key}' must be a torch.Tensor"
                )
            if not bool(torch.all(torch.isfinite(boundary_state))):
                raise ValueError("Diffusion boundary_state must contain only finite values")
            if self.non_negative and bool(torch.any(boundary_state < 0.0)):
                raise ValueError("Diffusion boundary_state must be non-negative")

        return source_term, boundary_state

    def _has_tracer_state(self, state: Mapping[str, Any]) -> bool:
        """Return True when a mapping contains the declared tracer state key."""

        return self.tracer_key in state

    def _quadratic_storage(self, concentration: Tensor) -> Tensor:
        """Compute the quadratic storage functional for structural audit."""

        # References:
        #   - Crank (1975), The Mathematics of Diffusion.
        #   - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        # Mathematical expression:
        #   H_h(c_h) = 0.5 * c_h.T c_h, e_h = delta H_h / delta c_h = c_h.
        # Parameter meanings:
        #   c_h is the tracer concentration cochain in state units.
        return 0.5 * torch.sum(concentration * concentration)

    def _diffusion_dissipation_rate(
        self, concentration: Tensor, boundary_state: Optional[Tensor] = None,
    ) -> Tensor:
        """Audit quadratic-norm loss on the production finite-volume faces."""

        # References:
        #   - Crank (1975), The Mathematics of Diffusion.
        #   - Patankar (1980), Numerical Heat Transfer and Fluid Flow.
        # Mathematical expression:
        #   H0 = sum_i(c_i**2)/2; sigma0 = sum_f K_f*(Delta c_f/dx_k)**2.
        #   Include each periodic seam once. Dirichlet boundary faces add
        #   2*K_edge*((c_edge-c_boundary)/dx_k)**2 (half-cell distance).
        #   Linear extrapolation adds K_edge*(Delta c_edge/dx_k)**2/2.
        # Parameter meanings:
        #   c is concentration, K_f is the shared harmonic face coefficient,
        #   dx_k is cell width, and boundary_state is the physical face value.
        # Units: c [kg m-3], K [m2 s-1], sigma0 [kg2 m-6 s-1].
        # Domain/constraints: uniform cell-centered axes with at least two
        #   cells, finite positive configured K, and the production boundary
        #   closure. This is a level-3 norm audit, not thermodynamic entropy.
        # Sign convention: dH0/dt = -sigma0 - sum_boundary(c_face*j_out/dx_k)
        #   without sources; positive sigma0 denotes internal norm dissipation.
        if not concentration.is_floating_point():
            raise TypeError("concentration must use a floating-point dtype")
        if self.non_negative and bool(torch.any(concentration < 0.0)):
            raise ValueError("FickianDiffusion requires a non-negative concentration state")
        boundary = None
        if self.boundary_method == "dirichlet":
            if boundary_state is None:
                raise ValueError("Dirichlet diffusion requires explicit boundary_state")
            if not isinstance(boundary_state, Tensor):
                raise TypeError("Diffusion boundary_state must be a torch.Tensor")
            boundary = boundary_state.to(device=concentration.device, dtype=concentration.dtype)
            if not bool(torch.all(torch.isfinite(boundary))):
                raise ValueError("Diffusion boundary_state must contain only finite values")
            if self.non_negative and bool(torch.any(boundary < 0.0)):
                raise ValueError("Diffusion boundary_state must be non-negative")
            try:
                boundary = torch.broadcast_to(boundary, concentration.shape)
            except RuntimeError as exc:
                raise ValueError("boundary_state must be broadcast-compatible with concentration") from exc
        elif boundary_state is not None:
            raise ValueError("boundary_state is only supported for Dirichlet diffusion")

        axis_diffusivities = self._resolve_axis_diffusivities(concentration)
        dissipation = torch.zeros((), device=concentration.device, dtype=concentration.dtype)
        spatial_axes = list(range(-self.spatial_dims, 0))
        for idx, axis in enumerate(spatial_axes):
            if concentration.size(axis) < 2:
                raise ValueError("Diffusion requires at least two cells along each axis")
            spacing = torch.as_tensor(
                float(self.grid_spacing[idx]),
                device=concentration.device,
                dtype=concentration.dtype,
            )
            gradient = torch.diff(concentration, dim=axis) / spacing
            coefficient = axis_diffusivities[idx]
            face_diffusivity = harmonic_mean(
                coefficient.narrow(axis, 0, coefficient.size(axis) - 1),
                coefficient.narrow(axis, 1, coefficient.size(axis) - 1),
            )
            dissipation = dissipation + torch.sum(face_diffusivity * gradient * gradient)
            if self.boundary_method == "periodic":
                seam_gradient = (concentration.select(axis, 0) - concentration.select(axis, -1)) / spacing
                seam_coefficient = harmonic_mean(coefficient.select(axis, 0), coefficient.select(axis, -1))
                dissipation = dissipation + torch.sum(seam_coefficient * seam_gradient.square())
            elif boundary is not None:
                for edge in (0, -1):
                    jump = (concentration.select(axis, edge) - boundary.select(axis, edge)) / spacing
                    dissipation = dissipation + torch.sum(2.0 * coefficient.select(axis, edge) * jump.square())
            elif self.boundary_method == "extrapolate":
                for edge in (0, -1):
                    edge_gradient = gradient.select(axis, edge)
                    dissipation = dissipation + torch.sum(0.5 * coefficient.select(axis, edge) * edge_gradient.square())
        return dissipation

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Use the canonical finite-volume mass ledger for local validation.

        References:
            Patankar (1980), Numerical Heat Transfer and Fluid Flow.
        Mathematical expression:
            Delta M - dt integral(S dV) + dt sum_f(j_out,f A_f) = 0.
        Parameter meanings:
            M is stored tracer mass, S is the volumetric source rate, j_out,f
            is the outward-positive face flux, and dt is the step duration.
        Units:
            M [kg], S [kg m-3 s-1], j_out,f [kg m-2 s-1], dt [s].
        Open/closed system route:
            Dirichlet and extrapolate exchange is routed through the explicit
            diffusive boundary flux; closed boundaries have zero port flow.
        """

        del input_state
        residual = output_updates.get("diagnostics.conservation_error")
        if residual is None:
            return {}
        if isinstance(residual, UpdateItem):
            residual = residual.value
        if not isinstance(residual, Tensor):
            raise TypeError("diagnostics.conservation_error must be a torch.Tensor")
        if not bool(torch.all(torch.isfinite(residual))):
            raise ValueError("diagnostics.conservation_error must be finite")
        magnitude = float(torch.max(torch.abs(residual)).detach().item())
        return {"mass": magnitude}

    def _ensure_batch_dimension(self, field: Tensor) -> Tuple[Tensor, int]:
        squeeze_dims = 0
        field_out = field
        missing_dims = max(0, self.spatial_dims + 1 - field_out.dim())
        for _ in range(missing_dims):
            field_out = field_out.unsqueeze(0)
            squeeze_dims += 1
        if self.spatial_dims > 1:
            spatial_shape = field_out.shape[-self.spatial_dims:]
            if any(size < 2 for size in spatial_shape):
                raise ValueError(
                    "Concentration tensor must contain at least two cells per spatial dimension."
                )
        return field_out, squeeze_dims

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _extract_concentration(self, state: StateDict) -> Tensor:
        if self.tracer_key not in state:
            raise ValueError(
                f"State must contain '{self.tracer_key}' for diffusion"
            )
        concentration = state[self.tracer_key]
        if not isinstance(concentration, Tensor):
            raise TypeError(
                f"State variable '{self.tracer_key}' must be a torch.Tensor"
            )
        concentration = concentration.to(self.device)
        if not torch.all(torch.isfinite(concentration)):
            raise ValueError("Fickian diffusion concentration must contain only finite values.")
        if concentration.dim() < self.spatial_dims:
            raise ValueError(
                f"Concentration tensor must have at least {self.spatial_dims} spatial dimensions; "
                f"received shape {tuple(concentration.shape)}."
            )
        return concentration

    def _resolve_axis_diffusivities(self, concentration: Tensor) -> List[Tensor]:
        """Use the production operator's scalar/cell/directional normalization."""
        tensor = self.get_diffusivity().to(device=concentration.device, dtype=concentration.dtype)
        return list(
            _normalize_directional_diffusivity(
                tensor, concentration, self.spatial_dims, self.diffusivity_layout
            )
        )

    def boundary_conditions(
        self, boundary_name: str, field_values: Tensor, normal_vector: Tensor,
        boundary_type: Optional[str] = None, boundary_value: Optional[float] = None,
        use_bc_manager: bool = True,
    ) -> Tensor:
        """Return the contract flux for a closed diffusive boundary.

        References:
          - Crank (1975), The Mathematics of Diffusion.
        Mathematical expression:
          j_out = -K grad(c) dot n = 0 for zero-Neumann closure.
        Parameter meanings:
          field_values is c [state_unit], normal_vector is the outward normal
          [1], and boundary_value is an optional prescribed outward flux.
        Units: returned zero has the boundary flux support of field_values.
        Domain/constraints: only the configured closed FV closure is queried;
          this query does not install alternative boundary-manager rules.
        Sign convention: positive flux is outward.
        Open/closed system route:
          Dirichlet and extrapolated exchange requires the explicit forward
          boundary state and is emitted by the canonical boundary flux port.
        """

        _ = boundary_name, normal_vector, use_bc_manager
        requested = self.boundary_method if boundary_type is None else boundary_type.strip().lower()
        zero_flux_types = {"zero", "neumann", "open"}
        if requested != self.boundary_method and not {requested, self.boundary_method} <= zero_flux_types:
            raise ValueError("Requested boundary closure differs from the configured forward boundary")
        if boundary_value is not None and (
            not math.isfinite(float(boundary_value)) or float(boundary_value) != 0.0
        ):
            raise RuntimeError("Nonzero boundary exchange requires the explicit forward boundary state and flux ledger")
        if self.boundary_method in {"zero", "neumann", "periodic", "open"}:
            return torch.zeros_like(field_values)
        raise RuntimeError(
            "Open diffusive boundary flux is available only from forward() "
            "with the explicit boundary state and oriented port ledger"
        )

    def _operator_spacing(self) -> Union[float, Tuple[float, ...]]:
        if self.spatial_dims == 1:
            return float(self.grid_spacing[0])
        return tuple(float(spacing) for spacing in self.grid_spacing[: self.spatial_dims])

def create_isotropic_diffusion(
    spatial_dims: int,
    diffusivity: float,
    grid_spacing: Optional[Union[float, Sequence[float]]] = None,
    **kwargs: Any,
) -> FickianDiffusion:
    """
    Factory for isotropic diffusion with scalar diffusivity.
    """
    return FickianDiffusion(
        spatial_dims=spatial_dims,
        diffusivity=diffusivity,
        grid_spacing=grid_spacing,
        **kwargs,
    )


def create_anisotropic_diffusion(
    spatial_dims: int,
    diffusivity_components: Sequence[float],
    grid_spacing: Optional[Union[float, Sequence[float]]] = None,
    **kwargs: Any,
) -> FickianDiffusion:
    """
    Factory for anisotropic diffusion with directional diffusivity components.
    """
    component_count = len(diffusivity_components)
    if component_count != spatial_dims:
        raise ValueError(
            f"Need {spatial_dims} diffusivity components for {spatial_dims}D diffusion, "
            f"got {component_count}."
        )
    tensor = torch.tensor(diffusivity_components)
    return FickianDiffusion(
        spatial_dims=spatial_dims,
        diffusivity=tensor,
        grid_spacing=grid_spacing,
        diffusivity_layout="directional",
        **kwargs,
    )


__all__ = [
    'FickianDiffusion',
    'create_isotropic_diffusion',
    'create_anisotropic_diffusion',
]
