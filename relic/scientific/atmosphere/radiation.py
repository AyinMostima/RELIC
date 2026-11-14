'\nAtmospheric Radiation Process Module for RELIC\n\nThis module implements comprehensive radiation transfer calculations including \nshortwave and longwave radiation balance at the surface for ecosystem modeling.\nProvides net radiation (Rn) and component fluxes for coupling with \nevapotranspiration, energy balance, and other atmospheric processes.\n\nPhysical Framework:\n    Surface radiation balance: R_net = SW_net + LW_net\n    SW_net = (1 - albedo) * SW_down\n    LW_net = epsilon * LW_down - epsilon * sigma * T_sfc^4\n    LW_up = LW_emitted + LW_reflected; LW_reflected = (1 - epsilon) * LW_down\n\nMathematical Expressions:\n    1. Clear-sky shortwave: SW_down = SW_potential * transmission_coeff * cos(zenith)\n    2. Cloud modification: SW_down = SW_clear * (1 - cloud_fraction * cloud_factor)\n    3. Atmospheric longwave (Brutsaert): LW_down = epsilon_atm * sigma * T_air^4\n    4. Atmospheric emissivity: epsilon_atm = 1.24 * (e_a / T_air)^(1/7)\n    5. Net radiation: R_net = SW_net + LW_net\n\nReferences:\n    - Allen, R.G. et al. (1998). FAO-56: Crop evapotranspiration guidelines\n    - Brutsaert, W. (1975). On a derivable formula for long-wave radiation\n    - Campbell, G.S. & Norman, J.M. (1998). Environmental Biophysics\n    - Monteith, J.L. & Unsworth, M.H. (2013). Principles of Environmental Physics\n\nScientific Validation:\n    - Energy conservation: SW_in = SW_absorbed + SW_reflected\n    - Physical bounds: 0 <= albedo <= 1, 0 <= emissivity <= 1\n    - Diurnal cycle: SW_down = 0 during night (cos_zenith <= 0)\n    - Clear-sky limits: cloud_fraction = 0 gives maximum SW transmission\n\nUnits (CF-1.8 Compliance):\n    - Radiation fluxes: W m-2\n    - Temperature: K\n    - Angles: radians\n    - Emissivity/albedo: dimensionless [0-1]\n'
import math
import torch
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, TypedDict, Union
from ...computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from ...computation.operators.iphs import PortSpecIPHS
from ...core.functor import (
    PhysicalFieldFunctor,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    DependencyList,
    TemporalScale,
)
from ...core.state import StateWriteMode, UpdateItem, UpdateMode
from ...core.abstractions import StateDict, FluxDict, CoordinateTensor, FieldTensor, GradientTensor, EnvironmentDict
from ...utils.unified_response_functions import validate_unity_range, validate_temperature_range

# Input plausibility domains of the radiation temperatures [K]. They reject only non-terrestrial values: the
# Stefan-Boltzmann longwave terms and the clear-sky emissivity hold at every positive temperature.
# References: WMO Archive of Weather and Climate Extremes (lowest recorded air temperature -89.2 degC, Vostok,
#   1983; highest 56.7 degC, Furnace Creek, 1913); Mildrexler et al. (2011), Bull. Amer. Meteor. Soc. 92,
#   855-860 (highest satellite land skin temperature 70.7 degC, Lut desert, 2005).
# Values: air [183.15, 333.15] K and surface [183.15, 353.15] K, each a few kelvin outside the record.
# Before 2026-11-05 both used the generic default [233.15, 333.15] K of validate_temperature_range, which
#   refused boreal and continental winter daily means on vegetated land (global G050 admission, paper 3).
RADIATION_AIR_TEMPERATURE_DOMAIN_K = (183.15, 333.15)
RADIATION_SURFACE_TEMPERATURE_DOMAIN_K = (183.15, 353.15)
from ...utils.physical_constants import get_physical_constant
from ...core.feature_flags import feature_manager
from ..hydrosphere.utils.psychrometrics import compute_saturation_vapor_pressure
from .radiation_energy_conservation import RadiationEnergyConservationValidator
from ...contracts import load_contract_registry


class _ResolvedRadiationInputs(TypedDict):
    """Typed container for validated forward inputs."""
    air_temp: torch.Tensor
    sw_potential: torch.Tensor
    cos_zenith: torch.Tensor
    surface_temp: torch.Tensor
    vapor_pressure: torch.Tensor
    surface_albedo: torch.Tensor
    surface_emissivity: torch.Tensor
    cloud_fraction: torch.Tensor


def _scalar_from_tensor(value: torch.Tensor) -> float:
    """Convert a non-empty scalar-like tensor to a Python float."""
    detached = value.detach()
    if detached.numel() == 0:
        raise ValueError('Cannot extract a scalar from an empty tensor')
    return float(detached.reshape(-1)[0].tolist())


class RadiationField(PhysicalFieldFunctor):
    """
    Physical Field Implementation for Atmospheric Radiation.
    
    This class implements the PhysicalField abstract interface for radiation
    data, providing spatially distributed radiation field values with proper
    interpolation, gradient calculation, and StateManager-owned snapshots.
    Designed to be fully consistent with the four-pillar architecture.
    
    Mathematical Framework:
        Field representation: R(x,y,t) where R is radiation flux [W m-2]
        Spatial interpolation: R_interp(x) = sum_i w_i(x) R(x_i)
        Gradient calculation: nablaR = (partialR/partialx, partialR/partialy) [W m^-3]
        
    Key Features:
        - Canonical inverse-distance spatial interpolation
        - Automatic gradient computation via PyTorch autograd
        - Time-dependent state snapshot projection
        - Conservation-aware diagnostics
        - Physical constraint enforcement
    
    References:
        - Brutsaert, W. (1975). On a derivable formula for long-wave radiation
        - Campbell & Norman (1998). Environmental Biophysics, Ch. 10-11
        - Shepard (1968). Inverse-distance interpolation
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    # Coordinate dimensions of the field's spatial query.
    SPATIAL_DIMS = 2

    def __init__(
        self,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ):
        """
        Initialize radiation field with physics-based parameters.

        The field is two-dimensional (``SPATIAL_DIMS``); its identity is the
        ``radiation_field`` output and its units come from that variable's
        state contract.

        Parameters:
            device : Optional[torch.device]
                Computation device (CPU/GPU)
        """
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise TypeError(
                "RadiationField uses the canonical radiation_field "
                f"contract; unsupported constructor arguments: {unsupported}"
            )
        field_name = "radiation_field"
        # Units of the field are read from its state contract (ContractRegistry
        # SSOT), never declared a second time here.
        field_contract = load_contract_registry().get_contract(field_name)
        if field_contract is None:
            raise KeyError(f"no state contract declares {field_name!r}")
        field_units = str(field_contract.udunits)
        metadata = FunctorMetadata(
            name=field_name,
            version="1.0.0",
            description="Atmospheric radiation field container and interpolator",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "R_out = clamp(R_state, R_min, R_max); "
                "nabla R = (partial R/partial x, partial R/partial y)"
            ),
            references=[
                "Brutsaert, W. (1975). On a derivable formula for long-wave radiation from clear skies.",
                "Campbell, G.S. and Norman, J.M. (1998). An Introduction to Environmental Biophysics."
            ],
            parameters={
                "atmosphere.radiation.field_bounds.generic_min_radiation": (
                    "Lower physical projection bound of the field [W m-2]"
                ),
                "atmosphere.radiation.field_bounds.generic_max_radiation": (
                    "Upper physical projection bound of the field [W m-2]"
                ),
                "atmosphere.radiation.field_interpolation.idw_power": (
                    "Inverse-distance exponent [dimensionless]"
                ),
                "atmosphere.radiation.field_interpolation.coincident_tolerance": (
                    "Coincident-sample distance tolerance [m]"
                ),
            },
            input_variables=[
                "grid_coordinates",
                "radiation_field",
                "source_coordinates",
            ],
            output_variables=[
                "radiation_field",
                "diagnostics.radiation_field.gradient",
                "diagnostics.radiation_field.clipping_residual",
                "diagnostics.radiation_field.query_clipping_residual",
            ],
            conservation_laws=["energy"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
        )
        super().__init__(
            name=field_name,
            metadata=metadata,
            device=device,
        )
        self._field_units = field_units
        self.optional_input_variables = ('source_coordinates',)
        self._field_values: Optional[torch.Tensor] = None
        self._grid_coordinates: Optional[torch.Tensor] = None
        self._time_series: Optional[torch.Tensor] = None
        self._last_clipping_residual: Optional[torch.Tensor] = None
        self.interpolation_method = 'inverse_distance'
        self.boundary_condition = 'zero_gradient'
        # Projection bounds of the field (forward and get_value_at clamp to them and route
        # the clipped part to the clipping-residual diagnostics); the field itself is
        # StateManager state or supplied to initialize_field, never a parameter default.
        self.min_radiation = self.get_parameter_value(
            'field_bounds.generic_min_radiation',
            namespace='atmosphere.radiation',
            required=True,
        )
        self.max_radiation = self.get_parameter_value(
            'field_bounds.generic_max_radiation',
            namespace='atmosphere.radiation',
            required=True,
        )
        self._idw_power = self.get_parameter_value(
            'field_interpolation.idw_power',
            namespace='atmosphere.radiation',
            required=True,
        )
        self._coincident_tolerance = self.get_parameter_value(
            'field_interpolation.coincident_tolerance',
            namespace='atmosphere.radiation',
            required=True,
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Return the exact ParameterManager bindings used by this field."""
        return (
            "atmosphere.radiation.field_bounds.generic_min_radiation",
            "atmosphere.radiation.field_bounds.generic_max_radiation",
            "atmosphere.radiation.field_interpolation.idw_power",
            "atmosphere.radiation.field_interpolation.coincident_tolerance",
        )

    def get_dependencies(self) -> DependencyList:
        """Explicitly declare no upstream dependencies for the radiation field container."""
        return self._merge_declared_dependencies([])

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Audit the exact bounded projection owned by this field container."""
        if 'radiation_field' not in input_state:
            raise KeyError('radiation_field is required for the field energy audit')
        if 'radiation_field' not in output_updates:
            raise KeyError('radiation_field output is required for the field energy audit')
        state_field = input_state['radiation_field']
        output_field = output_updates['radiation_field']
        if hasattr(output_field, 'value'):
            output_field = output_field.value
        if not isinstance(state_field, torch.Tensor) or not isinstance(
            output_field,
            torch.Tensor,
        ):
            raise TypeError('Radiation field energy audit requires tensor fields')

        # References: Campbell and Norman (1998), bounded radiation fields.
        # Mathematical expression: residual = R_out - clamp(R_state, R_min, R_max).
        # Parameter meanings: R_state and R_out are radiative flux fields [W m-2];
        # R_min and R_max are ParameterManager-backed field bounds [W m-2].
        expected = torch.clamp(
            state_field.to(device=output_field.device, dtype=output_field.dtype),
            min=self.min_radiation,
            max=self.max_radiation,
        )
        residual = output_field - expected
        if not torch.isfinite(residual).all():
            raise ValueError('Radiation field energy audit residual is non-finite')
        magnitude = _scalar_from_tensor(torch.max(torch.abs(residual)))
        return {'energy': magnitude}

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare radiation field cochains for Phase 8 audit.

        References:
          - Campbell and Norman (1998), radiation fields in environmental physics.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          R_h in C0(K) and grad R_h in C1(K), with R_t = M_h e_h + G_h u_h.
        Parameter meanings:
          R_h is the radiation flux field, grad R_h is the spatial radiation
          gradient, and G_h u_h represents boundary or source forcing.
        """

        if self._field_values is None:
            raise ValueError(
                "RadiationField declares its cochain layout from the initialized field; "
                "call initialize_field first."
            )
        n_values = int(self._field_values.numel())
        n_gradient = max(n_values * self.SPATIAL_DIMS, 1)
        return (
            CochainSpaceSpec(
                complex_id="atmosphere_radiation_field",
                degree=0,
                primality="primal",
                n_dofs=n_values,
                units=self._field_units,
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="radiation_flux_field",
                orientation_convention="positive_downward_flux",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="atmosphere_radiation_field",
                degree=1,
                primality="primal",
                n_dofs=n_gradient,
                units="W m-3",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="radiation_flux_gradient",
                orientation_convention="positive_spatial_increase",
                boundary_role="boundary",
            ),
        )

    def build_hodge_contract(
        self,
        n_dofs: int,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build a positive radiation-field metric Hodge contract.

        References:
          - Campbell and Norman (1998), surface radiation balance.
          - Desbrun et al. (2005), discrete Hodge star operators.
        Mathematical expression:
          M_h = diag(Delta R_max), Delta R_max = R_max - R_min, M_h >= 0.
        Parameter meanings:
          R_min and R_max are the ParameterManager-backed projection bounds
          atmosphere.radiation.field_bounds.generic_min_radiation and
          generic_max_radiation.
        Units:
          R_min, R_max and Delta R_max in W m-2.
        Domain/constraints:
          R_max > R_min, otherwise ValueError; the span is the positive metric.
        Sign convention:
          radiation fluxes are positive downward into the surface; Delta R_max >= 0.
        """

        n_local = int(n_dofs)
        if n_local < 1:
            raise ValueError("Radiation field Hodge degrees of freedom must be positive.")
        min_value = _scalar_from_tensor(torch.as_tensor(self.min_radiation))
        max_value = _scalar_from_tensor(torch.as_tensor(self.max_radiation))
        radiation_span = max_value - min_value
        if radiation_span <= 0.0:
            raise ValueError("Radiation field bounds must define a positive Hodge span.")
        values = torch.full(
            (n_local,),
            radiation_span,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        field_space = CochainSpaceSpec(
            complex_id="atmosphere_radiation_field",
            degree=0,
            primality="primal",
            n_dofs=n_local,
            units=self._field_units,
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="radiation_flux_field",
            orientation_convention="positive_downward_flux",
        )
        effort_space = CochainSpaceSpec(
            complex_id="atmosphere_radiation_field",
            degree=0,
            primality="dual",
            n_dofs=n_local,
            units=self._field_units,
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="radiation_field_effort",
            orientation_convention="positive_downward_flux",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_radiation_field_hodge",
            domain=field_space,
            codomain=effort_space,
            matrix=torch.diag(values),
            parameter_keys=self.required_parameters(),
            metric_keys=self.required_parameters(),
            material_law="bounded_radiation_field_metric",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build radiation field Hodge terms for Phase 8 assembly."""

        # The Hodge acts on the declared state layout unless the complex bundle
        # declares its own degrees of freedom.
        bundle_dofs = getattr(complex_bundle, "n_dofs", None) if complex_bundle is not None else None
        n_dofs = self.declared_forms()[0].n_dofs if bundle_dofs is None else bundle_dofs
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a structural radiation-field energy audit view."""

        field_value = state["radiation_field"] if "radiation_field" in state else self._field_values
        if field_value is None:
            return {
                "quadratic_storage_audit": "H_audit = 0.5 * R_h.T R_h",
                "effort_audit": "e_h = R_h",
            }
        tensor = field_value if isinstance(field_value, torch.Tensor) else torch.as_tensor(field_value, device=self.device)
        radiation = tensor.to(device=self.device, dtype=torch.float32).reshape(-1)
        # References: Campbell and Norman (1998), radiation energy balance.
        # Mathematical expression: H_audit = 0.5 * R_h.T R_h.
        # Parameter meanings: R_h is the ParameterManager-bounded radiation field.
        return {
            "quadratic_storage_audit": 0.5 * torch.sum(radiation * radiation),
            "effort_audit": radiation,
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return non-negative structural radiation-field dissipation terms."""

        field_value = state["radiation_field"] if "radiation_field" in state else self._field_values
        if field_value is None:
            return {
                "dissipation_rate": "sigma_h = R_h.T M_h R_h",
                "thermodynamic_force": "R_h",
            }
        tensor = field_value if isinstance(field_value, torch.Tensor) else torch.as_tensor(field_value, device=self.device)
        radiation = tensor.to(device=self.device, dtype=torch.float32).reshape(-1)
        hodge = self.build_hodge_contract(n_dofs=radiation.numel(), strict=True).matrix
        # References: IPHS dissipative field form and DEC Hodge passivity.
        # Mathematical expression: sigma_h = R_h.T M_h R_h >= 0.
        # Parameter meanings: M_h is assembled from radiation field bound parameters.
        return {
            "dissipation_rate": radiation @ hodge @ radiation,
            "thermodynamic_force": radiation,
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for this reduced radiation field."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive radiation-field metric."""

        hodge = self.build_hodge_contract(
            n_dofs=self.declared_forms()[0].n_dofs, strict=True
        ).matrix
        if bool(torch.any(hodge < 0.0)):
            raise ValueError("Radiation field mobility must be non-negative in M_h.")
        return {
            "state_vars": ("radiation_field",),
            "J": torch.zeros_like(hodge),
            "M": hodge,
            "ports": self.ports(),
            "open_system_route": "port",
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the radiation field boundary flux port.

        References:
          - Campbell and Norman (1998), surface radiation boundary fluxes.
          - van der Schaft and Maschke (1995), port-Hamiltonian boundary ports.
        Mathematical expression:
          P_port = e_R q_R, with q_R representing boundary radiation flux.
        Parameter meanings:
          e_R is the radiation field effort and q_R is the boundary flux flow.
        """

        effort_space = CochainSpaceSpec(
            complex_id="atmosphere_radiation_field",
            degree=0,
            primality="primal",
            n_dofs=1,
            units=self._field_units,
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="radiation_field_boundary_effort",
            orientation_convention="positive_downward_flux",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="atmosphere_radiation_field",
            degree=0,
            primality="dual",
            n_dofs=1,
            units=self._field_units,
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="radiation_boundary_flux",
            orientation_convention="positive_into_domain",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="radiation_field_boundary_port",
                effort_var="radiation_field",
                flow_var="radiation_boundary_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units=self._field_units,
                positive_power_convention="into_system",
                port_kind="radiation",
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

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> Dict[str, Any]:
        """Return the finite StateManager-owned radiation field and diagnostics.

        References:
            Campbell and Norman (1998), environmental radiation fields.
        Mathematical expression:
            R_out = clamp(R_state, R_min, R_max).
        Parameter meanings:
            R_state is the required state field [W m-2]; R_min and R_max are
            ParameterManager-backed physical bounds [W m-2].
        """
        # References: Campbell and Norman (1998), environmental radiation fields.
        # Mathematical expression: R_out = clamp(R_state, R_min, R_max).
        # Parameter meanings: R_state [W m-2]; R_min and R_max [W m-2].
        unsupported = sorted(set(kwargs) - {'return_gradients'})
        if unsupported:
            raise TypeError(
                'RadiationField.forward accepts only state-driven inputs; '
                f"unsupported keyword arguments: {', '.join(unsupported)}"
            )
        if not isinstance(dt, (int, float)):
            raise TypeError('dt must be a finite positive number')
        if not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError('dt must be positive and finite')

        return_gradients = kwargs.get('return_gradients', False)
        if not isinstance(return_gradients, bool):
            raise TypeError('return_gradients must be a bool')
        current_field = self._synchronize_field_from_state(state)
        self._last_clipping_residual = None
        unclamped_output = current_field.detach().clone()
        clipped_output = torch.clamp(
            unclamped_output,
            min=self.min_radiation,
            max=self.max_radiation,
        )
        residual = clipped_output - unclamped_output
        if not bool(torch.isfinite(clipped_output).all()):
            raise ValueError('radiation_field output must contain only finite values')
        output: Dict[str, Any] = {
            'radiation_field': UpdateItem(
                value=clipped_output,
                mode=UpdateMode.REPLACE,
                unit='W m-2',
                per_timestep=True,
                source_functor=self.name,
            )
        }
        if return_gradients:
            query_coords = self._resolve_query_coordinates(state)
            if not isinstance(query_coords, torch.Tensor):
                raise TypeError('radiation gradient coordinates must be a torch.Tensor')
            if not bool(torch.isfinite(query_coords).all()):
                raise ValueError('radiation gradient coordinates must be finite')
            gradients = self.compute_gradient(query_coords)
            if not bool(torch.isfinite(gradients).all()):
                raise ValueError('radiation_gradient output must contain only finite values')
            output['diagnostics.radiation_field.gradient'] = UpdateItem(
                value=gradients.detach().clone(),
                mode=UpdateMode.REPLACE,
                unit='W m-3',
                per_timestep=True,
                source_functor=self.name,
            )
        if self._last_clipping_residual is not None:
            query_residual = self._last_clipping_residual.to(
                device=residual.device,
                dtype=residual.dtype,
            )
            if bool(torch.any(query_residual != 0.0)):
                output['diagnostics.radiation_field.query_clipping_residual'] = UpdateItem(
                    value=query_residual.detach().clone(),
                    mode=UpdateMode.REPLACE,
                    unit='W m-2',
                    per_timestep=True,
                    source_functor=self.name,
                )
        if bool(torch.any(residual != 0.0)):
            output['diagnostics.radiation_field.clipping_residual'] = UpdateItem(
                value=residual.detach().clone(),
                mode=UpdateMode.REPLACE,
                unit='W m-2',
                per_timestep=True,
                source_functor=self.name,
            )
        return output

    def _as_device_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.tensor(value, dtype=torch.float32, device=self.device)

    def _synchronize_field_from_state(self, state: StateDict) -> torch.Tensor:
        """Synchronize interpolation storage from required finite state tensors."""
        if 'grid_coordinates' not in state:
            raise KeyError('grid_coordinates')
        if 'radiation_field' not in state:
            raise KeyError('radiation_field')
        grid_coordinates = state['grid_coordinates']
        radiation_field = state['radiation_field']
        if not isinstance(grid_coordinates, torch.Tensor):
            raise TypeError('grid_coordinates must be a torch.Tensor')
        if not isinstance(radiation_field, torch.Tensor):
            raise TypeError('radiation_field must be a torch.Tensor')
        if not bool(torch.isfinite(grid_coordinates).all()):
            raise ValueError('grid_coordinates must contain only finite values')
        if not bool(torch.isfinite(radiation_field).all()):
            raise ValueError('radiation_field must contain only finite values')
        if grid_coordinates.dim() != 2 or grid_coordinates.size(1) != self.SPATIAL_DIMS:
            raise ValueError(
                f'grid_coordinates must have shape [n_points, {self.SPATIAL_DIMS}]'
            )
        if radiation_field.dim() != 1:
            raise ValueError('radiation_field must be one-dimensional')
        if radiation_field.size(0) != grid_coordinates.size(0):
            raise ValueError(
                'radiation_field length must match the grid_coordinates point count'
            )
        if self._grid_coordinates is not None and grid_coordinates.shape != self._grid_coordinates.shape:
            raise ValueError('grid_coordinates shape does not match the initialized field grid')
        if self._field_values is not None and radiation_field.shape != self._field_values.shape:
            raise ValueError('radiation_field shape does not match the initialized field shape')
        self._grid_coordinates = grid_coordinates.to(self.device).detach().clone()
        self._field_values = radiation_field.to(self.device).clone()
        if not self._field_values.requires_grad:
            self._field_values.requires_grad_(True)
        return self._field_values

    def _resolve_query_coordinates(self, state: StateDict) -> Any:
        if 'source_coordinates' in state:
            return state['source_coordinates']
        if 'grid_coordinates' in state:
            return state['grid_coordinates']
        raise KeyError('source_coordinates or grid_coordinates is required')

    def initialize_field(self, grid_coordinates: torch.Tensor, initial_values: torch.Tensor) -> None:
        """
        Initialize the radiation field with spatial grid and initial values.

        Parameters:
            grid_coordinates : torch.Tensor
                Grid coordinate points
                Shape: [n_points, spatial_dims] for spatial coordinates
            initial_values : torch.Tensor
                Initial radiation values at grid points (required: the field has no
                parameter default)
                Shape: [n_points] with units [W m-2]
        """
        if grid_coordinates.dim() != 2 or grid_coordinates.size(1) != self.SPATIAL_DIMS:
            raise ValueError(f'Grid coordinates must have shape [n_points, {self.SPATIAL_DIMS}]')
        n_points = grid_coordinates.size(0)
        if initial_values.size(0) != n_points:
            raise ValueError('Initial values must match number of grid points')
        self._grid_coordinates = grid_coordinates.to(self.device)
        self._field_values = initial_values.to(self.device)
        self._field_values.requires_grad_(True)

    def get_value_at(self, coordinates: CoordinateTensor) -> FieldTensor:
        """
        Get radiation field values at specified coordinates using interpolation.
        
        Evaluates R(x,y) at arbitrary spatial locations using bilinear/bicubic
        interpolation. Critical for coupling between different spatial grids.
        
        Parameters:
            coordinates : CoordinateTensor
                Query coordinates
                Shape: [n_query, spatial_dims]
                Units: [m] for spatial coordinates
                
        Returns:
            FieldTensor: Interpolated radiation values
                Shape: [n_query]
                Units: [W m-2]
                
        Mathematical Expression:
            Shepard (1968) inverse-distance weighting (_inverse_distance_interpolate):
            w_i = d_i**(-p) / sum_j d_j**(-p), R(x) = sum_i w_i R_i, clamped to
            the field projection bounds
        """
        if not self.validate_coordinates(coordinates):
            raise ValueError('Invalid coordinate tensor dimensions or values')
        if self._field_values is None or self._grid_coordinates is None:
            raise RuntimeError('Radiation field not initialized. Call initialize_field() first.')
        coordinates = coordinates.to(self.device)
        interpolated_values = self._inverse_distance_interpolate(coordinates)
        clipped_values = torch.clamp(interpolated_values, min=self.min_radiation, max=self.max_radiation)
        self._last_clipping_residual = clipped_values - interpolated_values
        return clipped_values

    def _inverse_distance_interpolate(
        self,
        coordinates: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolate an irregular radiation field with normalized IDW weights.

        References:
            Shepard (1968), inverse-distance interpolation.
        Mathematical expression:
            w_i = d_i**(-p) / sum_j d_j**(-p), R_q = sum_i w_i R_i.
            A coincident query uses the exact source value instead.
        Parameter meanings:
            d_i is source-query distance [m], p is idw_power [-], and the
            coincident tolerance [m] selects exact samples.
        Open/closed system route:
            This is a closed spatial projection with no source or sink.
        """
        if self._grid_coordinates is None or self._field_values is None:
            raise RuntimeError('Radiation field not initialized for interpolation.')
        distances = torch.cdist(coordinates, self._grid_coordinates)
        tolerance = torch.as_tensor(
            self._coincident_tolerance,
            dtype=distances.dtype,
            device=distances.device,
        )
        power = torch.as_tensor(
            self._idw_power,
            dtype=distances.dtype,
            device=distances.device,
        )
        coincident = distances <= tolerance
        has_coincident = torch.any(coincident, dim=1, keepdim=True)
        coincident_weights = coincident.to(distances.dtype)
        coincident_weights = coincident_weights / torch.clamp(
            coincident_weights.sum(dim=1, keepdim=True),
            min=1.0,
        )
        safe_distances = torch.where(
            coincident,
            torch.ones_like(distances),
            distances,
        )
        idw_weights = torch.pow(safe_distances, -power)
        idw_weights = idw_weights / idw_weights.sum(dim=1, keepdim=True)
        weights = torch.where(has_coincident, coincident_weights, idw_weights)
        interpolated = torch.sum(weights * self._field_values.unsqueeze(0), dim=1)
        return interpolated

    def compute_gradient(self, coordinates: CoordinateTensor) -> GradientTensor:
        """
        Compute spatial gradient of radiation field using automatic differentiation.
        
        Calculates nablaR = (partialR/partialx, partialR/partialy) using PyTorch's autograd system.
        Essential for radiation-driven transport and coupling processes.
        
        Parameters:
            coordinates : CoordinateTensor
                Query coordinates for gradient calculation
                Shape: [n_query, spatial_dims]
                
        Returns:
            GradientTensor: Radiation field gradients
                Shape: [n_query, spatial_dims]
                Units: [W m^-3] (flux gradient per distance)
                
        Mathematical Expression:
            Gradient: nablaR = (partialR/partialx, partialR/partialy)
            Using automatic differentiation: partialR/partialx = d(get_value_at)/d(coordinates)
        """
        if not self.validate_coordinates(coordinates):
            raise ValueError('Invalid coordinate tensor for gradient calculation')
        coordinates = coordinates.clone().detach().requires_grad_(True).to(self.device)
        field_values = self.get_value_at(coordinates)
        gradients = torch.zeros_like(coordinates)
        for i in range(coordinates.size(0)):
            if field_values[i].requires_grad:
                grad_outputs = torch.zeros_like(field_values)
                grad_outputs[i] = 1.0
                retain_graph = i < coordinates.size(0) - 1
                grad = torch.autograd.grad(
                    outputs=field_values[i],
                    inputs=coordinates,
                    retain_graph=retain_graph,
                    create_graph=True,
                    only_inputs=True,
                )[0]
                if grad is not None:
                    gradients[i] = grad[i]
        return gradients

    def interpolate(self, source_coordinates: CoordinateTensor, target_coordinates: CoordinateTensor) -> FieldTensor:
        """
        Interpolate radiation field from source to target coordinate system.

        Evaluates the inverse-distance query of get_value_at on the source
        support; it is not a conservative remap (radiation_field is an
        intensive flux density).
        
        Parameters:
            source_coordinates : CoordinateTensor
                Source grid coordinates
                Shape: [n_source, spatial_dims]
            target_coordinates : CoordinateTensor
                Target grid coordinates  
                Shape: [n_target, spatial_dims]

        The field has one scheme: the inverse-distance query of get_value_at on
        the source support (no method selection).

        Returns:
            FieldTensor: Interpolated field values at target coordinates
                Shape: [n_target]
                Units: [W m-2]
                
        Mathematical Expression:
            Shepard (1968) inverse-distance weighting on the source support:
            w_i = d_i**(-p) / sum_j d_j**(-p), R_new(x) = sum_i w_i R_old(x_i)
        """
        if not self.validate_coordinates(source_coordinates):
            raise ValueError('Invalid source coordinates')
        if not self.validate_coordinates(target_coordinates):
            raise ValueError('Invalid target coordinates')
        original_coords = self._grid_coordinates.clone() if self._grid_coordinates is not None else None
        original_values = self._field_values.clone() if self._field_values is not None else None
        if not torch.equal(source_coordinates, self._grid_coordinates):
            if self._field_values is None:
                raise RuntimeError('Radiation field values are not initialized')
            if source_coordinates.size(0) != self._field_values.size(0):
                raise ValueError(
                    'source_coordinates size must match current field value count'
                )
            # Temporarily remap source grid for interpolation query.
            self._grid_coordinates = source_coordinates.to(self.device)
        target_values = self.get_value_at(target_coordinates)
        if original_coords is not None:
            self._grid_coordinates = original_coords
        if original_values is not None:
            self._field_values = original_values
        return target_values

    def update_field(self, time_step: float, source_terms: EnvironmentDict, boundary_conditions: Optional[Dict[str, torch.Tensor]]=None) -> None:
        """Reject direct field mutation outside the StateManager transaction."""
        raise RuntimeError(
            'RadiationField is state-driven; update radiation_field through '
            'StateManager and call forward instead of update_field'
        )

class RadiationFunctor(PhysicalFieldFunctor):
    """
    Atmospheric radiation transfer functor for surface energy balance.
    
    Implements comprehensive radiation calculations including shortwave and
    longwave components with cloud effects, atmospheric emissivity, and
    surface properties. Provides net radiation for ecosystem energy balance.
    
    Key Features:
        - Clear-sky and cloudy-sky shortwave radiation
        - Atmospheric longwave emission using empirical formulas
        - Surface albedo and emissivity effects
        - Solar zenith angle dependencies
        - Energy conservation validation
        - Diurnal cycle handling
    
    Process Category: PHYSICAL_FIELD
    Execution Phase: MAIN_PROCESSES
    
    Input Variables:
        - cos_zenith [dimensionless]
        - incoming_shortwave_potential [W m-2] (extraterrestrial)
        - air_temperature [K]
        - surface_temperature [K]
        - relative_humidity [dimensionless]
        - surface_albedo [dimensionless]
        - surface_emissivity [dimensionless]
        - cloud_fraction [dimensionless]
        - leaf_area_index [m2 m-2] when canopy radiation is enabled
    
    Output Variables:
        - shortwave_down [W m-2]
        - longwave_down [W m-2] 
        - net_radiation [W m-2]
        - shortwave_net [W m-2]
        - longwave_net [W m-2]
        - ppfd [umol photons m-2 s-1] for every execution
        - fpar and apar_layer when canopy radiation is enabled
    
    Coupling Interfaces:
        - Evapotranspiration modules (consume net_radiation)
        - Surface energy balance (consume component fluxes)
        - Photosynthesis modules (consume canonical ppfd)
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    _CANONICAL_OUTPUT_UNITS = {
        'shortwave_down': 'W m-2',
        'longwave_down': 'W m-2',
        'net_radiation': 'W m-2',
        'shortwave_net': 'W m-2',
        'longwave_net': 'W m-2',
        'ppfd': 'umol photons m-2 s-1',
    }
    _DIAGNOSTIC_OUTPUT_UNITS = {
        'atmospheric_emissivity': 'dimensionless',
        'diagnostics.radiation.par_energy': 'W m-2',
        'diagnostics.radiation.par_direct': 'W m-2',
        'diagnostics.radiation.par_diffuse': 'W m-2',
        'diagnostics.radiation.ppfd_direct': 'umol photons m-2 s-1',
        'diagnostics.radiation.ppfd_diffuse': 'umol photons m-2 s-1',
        'diagnostics.radiation.diffuse_fraction': 'dimensionless',
        'diagnostics.radiation.longwave_up': 'W m-2',
        'diagnostics.radiation.longwave_emitted': 'W m-2',
        'diagnostics.radiation.longwave_reflected': 'W m-2',
        'diagnostics.radiation.surface_albedo_used': 'dimensionless',
        'diagnostics.radiation.surface_emissivity_used': 'dimensionless',
        'diagnostics.radiation.cos_zenith_used': 'dimensionless',
        'diagnostics.radiation.vapor_pressure_calculated': 'Pa',
        'diagnostics.radiation.shortwave_clipping_residual': 'W m-2',
        'diagnostics.radiation.emissivity_clipping_residual': 'dimensionless',
    }
    _CANOPY_OUTPUT_UNITS = {
        'fpar': 'dimensionless',
        'apar_layer': 'W m-2',
    }
    _CANOPY_DIAGNOSTIC_OUTPUT_UNITS = {
        'diagnostics.radiation.transmission_direct': 'W m-2',
        'diagnostics.radiation.transmission_diffuse': 'W m-2',
    }
    _CANOPY_SHARED_PARAMETER_SPECS = (
        ('erbs_low_threshold', 'dimensionless', None),
        ('erbs_high_threshold', 'dimensionless', None),
        ('erbs_low_coefficients', 'dimensionless', list),
        ('erbs_mid_coefficients', 'dimensionless', list),
        ('erbs_high_fraction', 'dimensionless', None),
        ('ppfd_conversion', 'umol J-1', None),
        ('par_fraction_of_sw', 'dimensionless', None),
    )
    _CANOPY_LAYER_PARAMETER_SPECS = (
        ('model_type', 'dimensionless', str),
        ('layers', 'dimensionless', int),
        ('k_b', 'dimensionless', None),
        ('k_d', 'dimensionless', None),
        ('lai_profile', 'dimensionless', str),
        ('beta_params', 'dimensionless', list),
    )

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(
        self,
        name: str = 'atmospheric_radiation',
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Initialize radiation functor with default parameters."""
        device = kwargs.pop('device', None)
        parameter_manager = kwargs.pop('parameter_manager', None)
        if kwargs:
            raise TypeError(
                'Unsupported RadiationFunctor initialization options: '
                + ', '.join(sorted(kwargs))
            )
        raw_config = config
        if raw_config is not None and not isinstance(raw_config, dict):
            raise TypeError('RadiationFunctor config must be a dictionary')
        config_dict: Dict[str, Any] = dict(raw_config or {})
        clean_settings: Dict[str, Any] = {}
        for setting_key in ('dependencies', 'strict_validation', 'strict_validation_errors'):
            if setting_key in config_dict:
                clean_settings[setting_key] = config_dict.pop(setting_key)
        enable_conservation_check = config_dict.pop('enable_conservation_check', True)
        if not isinstance(enable_conservation_check, bool):
            raise TypeError('enable_conservation_check must be a bool')
        self._enable_conservation_check = enable_conservation_check
        conservation_validator_config = config_dict.pop('conservation_validator', {})
        if not isinstance(conservation_validator_config, dict):
            raise TypeError('conservation_validator must be a dictionary')
        self._conservation_validator_config = conservation_validator_config
        if config_dict:
            raise ValueError(
                'Unsupported RadiationFunctor config keys: '
                + ', '.join(sorted(config_dict))
            )
        self._conservation_validator = None
        self._last_conservation_report: Optional[Dict[str, Any]] = None
        self._last_shortwave_clipping_residual: Optional[torch.Tensor] = None
        self._last_emissivity_clipping_residual: Optional[torch.Tensor] = None
        metadata = FunctorMetadata(
            name=name,
            version='1.0.0',
            description='Atmospheric radiation transfer for surface energy balance',
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                'R_net = SW_net + LW_net; '
                'SW_net = (1-alpha) * SW_down; '
                'LW_net = epsilon*LW_down - epsilon*sigma*T_sfc^4; '
                'LW_up = epsilon*sigma*T_sfc^4 + (1-epsilon)*LW_down; '
                'SW_down = SW_pot * tau * max(0,cos(theta)) * f_cloud; '
                'LW_down = epsilon_atm * sigma * T_air^4; '
                'epsilon_atm = C1 * (e_a/T_air)^(C2); '
                'PPFD = SW_down * f_PAR * c_PPFD'
            ),
            references=[
                'Allen, R.G. et al. (1998). FAO-56: Crop evapotranspiration guidelines',
                'Brutsaert, W. (1975). On a derivable formula for long-wave radiation from clear skies',
                'Campbell, G.S. & Norman, J.M. (1998). Environmental Biophysics',
                'Erbs, D.G. et al. (1982). Solar Energy 28, 293-302',
                'Monteith, J.L. & Unsworth, M.H. (2013). Principles of Environmental Physics',
                'Nicodemus (1965), Applied Optics 4, 767-775, doi:10.1364/AO.4.000767',
                'Noah-MP Technical Note, Eqs. 3.6.330 and 3.6.416-417',
            ],
            parameters={
                'atmosphere.radiation.atmospheric_transmittance': (
                    'Clear-sky shortwave transmittance [dimensionless]'
                ),
                'atmosphere.radiation.cloud_attenuation_factor': (
                    'Cloud shortwave attenuation [dimensionless]'
                ),
                'atmosphere.radiation.brutsaert_coefficient': (
                    'Brutsaert emissivity coefficient [dimensionless]'
                ),
                'atmosphere.radiation.brutsaert_exponent': (
                    'Brutsaert emissivity exponent [dimensionless]'
                ),
                'atmosphere.radiation.emissivity_clear_min': (
                    'Minimum clear-sky emissivity [dimensionless]'
                ),
                'atmosphere.radiation.emissivity_clear_max': (
                    'Maximum clear-sky emissivity [dimensionless]'
                ),
                'atmosphere.radiation.cloud_emissivity': (
                    'Cloud longwave emissivity [dimensionless]'
                ),
                'atmosphere.radiation.canopy.model_type': (
                    'Canopy radiation model type, read with canopy radiation enabled '
                    '[dimensionless]'
                ),
                'atmosphere.radiation.canopy.layers': (
                    'Number of canopy layers, read with canopy radiation enabled [dimensionless]'
                ),
                'atmosphere.radiation.canopy.k_b': (
                    'Direct-beam extinction coefficient, read with canopy radiation enabled '
                    '[dimensionless]'
                ),
                'atmosphere.radiation.canopy.k_d': (
                    'Diffuse extinction coefficient, read with canopy radiation enabled '
                    '[dimensionless]'
                ),
                'atmosphere.radiation.canopy.erbs_low_threshold': (
                    'Lower Erbs clearness-index threshold [dimensionless]'
                ),
                'atmosphere.radiation.canopy.erbs_high_threshold': (
                    'Upper Erbs clearness-index threshold [dimensionless]'
                ),
                'atmosphere.radiation.canopy.erbs_low_coefficients': (
                    'Erbs low-clearness polynomial coefficients [dimensionless]'
                ),
                'atmosphere.radiation.canopy.erbs_mid_coefficients': (
                    'Erbs middle-clearness polynomial coefficients [dimensionless]'
                ),
                'atmosphere.radiation.canopy.erbs_high_fraction': (
                    'Erbs high-clearness diffuse fraction [dimensionless]'
                ),
                'atmosphere.radiation.canopy.lai_profile': (
                    'Vertical LAI profile type, read with canopy radiation enabled '
                    '[dimensionless]'
                ),
                'atmosphere.radiation.canopy.beta_params': (
                    'Beta-profile coefficients (a, b), read with canopy radiation enabled '
                    '[dimensionless]'
                ),
                'atmosphere.radiation.canopy.ppfd_conversion': (
                    'PAR energy to PPFD conversion factor [umol J-1]'
                ),
                'atmosphere.radiation.canopy.par_fraction_of_sw': (
                    'PAR fraction of shortwave radiation [dimensionless]'
                ),
            },
            input_variables=[
                'air_temperature',
                'incoming_shortwave_potential',
                'cos_zenith',
                'surface_temperature',
                'relative_humidity',
                'surface_albedo',
                'surface_emissivity',
                'cloud_fraction',
            ],
            output_variables=(
                list(self._CANONICAL_OUTPUT_UNITS)
                + list(self._DIAGNOSTIC_OUTPUT_UNITS)
            ),
            conservation_laws=['energy'],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.MINUTE,
            coupling_interfaces=[
                'evapotranspiration',
                'surface_energy_balance',
                'photosynthesis',
            ],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )
        super().__init__(
            name=name,
            config=clean_settings,
            metadata=metadata,
            device=device,
            parameter_manager=parameter_manager,
        )
        self._radiation_params = self._load_core_parameters()
        self._canopy_enabled_at_initialization = self._canopy_execution_enabled()
        self._canopy_params = self._load_canopy_parameters()
        if self._canopy_enabled_at_initialization:
            self.metadata.input_variables.append('leaf_area_index')
            self.metadata.output_variables.extend(
                list(self._CANOPY_OUTPUT_UNITS)
                + list(self._CANOPY_DIAGNOSTIC_OUTPUT_UNITS)
            )
        self._stefan_boltzmann = self._load_physical_constants()
        self._standard_temperature = get_physical_constant('celsius_to_kelvin')
        self._tetens_base_pressure = get_physical_constant(
            'relic.physical.constants.psychrometrics.tetens_base_pressure'
        )
        self._tetens_coefficient_a = get_physical_constant(
            'relic.physical.constants.psychrometrics.tetens_coefficient_a'
        )
        self._tetens_coefficient_b = get_physical_constant(
            'relic.physical.constants.psychrometrics.tetens_coefficient_b'
        )
        self.device = self.device if getattr(self, 'device', None) is not None else self._infer_device()
        if device is not None:
            self.device = torch.device(device)

    def _get_conservation_validator(self) -> Optional[RadiationEnergyConservationValidator]:
        """Lazily build the conservation validator to avoid init-time hard dependencies."""
        if not self._enable_conservation_check or not feature_manager.is_enabled('radiation_conservation'):
            return None
        if self._conservation_validator is None:
            self._conservation_validator = RadiationEnergyConservationValidator(
                config=self._conservation_validator_config
            )
        return self._conservation_validator

    def _load_physical_constants(self) -> float:
        """Load the radiation physical constant used by the runtime."""
        return float(get_physical_constant('stefan_boltzmann_constant'))

    def _fetch_parameter(
        self,
        key: str,
        *,
        namespace: str,
        units: Optional[str],
        caster: Optional[Callable[[Any], Any]],
    ) -> Any:
        """Fetch a parameter from the unified parameter manager with casting."""
        value = self.get_parameter_value(
            key,
            namespace=namespace,
            units=units,
            source_order=('manager',),
            required=True,
        )
        return self._cast_parameter_value(value, caster)

    @staticmethod
    def _cast_parameter_value(value: Any, caster: Optional[Callable[[Any], Any]]) -> Any:
        """Cast parameter values, handling torch tensors and sequence types."""
        if caster is None:
            return value
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                scalar = value.detach().reshape(-1)[0].tolist()
                if caster is bool and not isinstance(scalar, bool):
                    raise TypeError('Boolean radiation parameters must use true or false')
                return caster(scalar)
            if caster in (list, tuple):
                sequence = value.detach().tolist()
                return sequence if caster is list else tuple(sequence)
            raise ValueError('Multi-element tensor cannot be cast to scalar parameter.')
        if caster is list:
            if isinstance(value, list):
                return value
            if isinstance(value, tuple):
                return list(value)
            return [value]
        if caster is tuple:
            if isinstance(value, tuple):
                return value
            if isinstance(value, list):
                return tuple(value)
            return (value,)
        if caster is bool and not isinstance(value, bool):
            raise TypeError('Boolean radiation parameters must use true or false')
        return caster(value)

    def _load_core_parameters(self) -> Dict[str, Any]:
        """Load core radiation parameters from ParameterManager."""
        namespace = 'atmosphere.radiation'
        specs: List[Tuple[str, str, Optional[Callable[[Any], Any]]]] = [
            ('atmospheric_transmittance', 'dimensionless', None),
            ('cloud_attenuation_factor', 'dimensionless', None),
            ('brutsaert_coefficient', 'dimensionless', None),
            ('brutsaert_exponent', 'dimensionless', None),
            ('emissivity_clear_min', 'dimensionless', None),
            ('emissivity_clear_max', 'dimensionless', None),
            ('cloud_emissivity', 'dimensionless', None),
        ]
        params: Dict[str, Any] = {}
        for key, units, caster in specs:
            params[key] = self._fetch_parameter(
                key,
                namespace=namespace,
                units=units,
                caster=caster,
            )
        return params

    def _load_canopy_parameters(self) -> Dict[str, Any]:
        """Load canopy radiation parameters from ParameterManager."""
        namespace = 'atmosphere.radiation.canopy'
        specs: List[Tuple[str, str, Optional[Callable[[Any], Any]]]] = list(
            self._CANOPY_SHARED_PARAMETER_SPECS
        )
        if self._canopy_enabled_at_initialization:
            specs.extend(self._CANOPY_LAYER_PARAMETER_SPECS)
        params: Dict[str, Any] = {}
        for key, units, caster in specs:
            params[key] = self._fetch_parameter(
                key,
                namespace=namespace,
                units=units,
                caster=caster,
            )

        if self._canopy_enabled_at_initialization:
            model_type = str(params["model_type"]).strip().lower()
            if model_type != "beer_lambert":
                raise ValueError(
                    "Unsupported canopy.model_type='{}'. Supported: 'beer_lambert'.".format(
                        model_type
                    )
                )
            params["model_type"] = model_type
            profile_type = str(params["lai_profile"]).strip().lower()
            if profile_type not in {"uniform", "beta"}:
                raise ValueError(
                    "Unsupported canopy.lai_profile='{}'. Supported: 'uniform', 'beta'.".format(
                        profile_type
                    )
                )
            params["lai_profile"] = profile_type
            beta_params = params["beta_params"]
            if not isinstance(beta_params, list) or len(beta_params) != 2:
                raise ValueError("canopy.beta_params must contain exactly two coefficients.")
            if any(float(value) <= 0.0 for value in beta_params):
                raise ValueError("canopy.beta_params coefficients must be positive.")
        low_threshold = self._coerce_float_parameter(
            params['erbs_low_threshold'],
            'erbs_low_threshold',
        )
        high_threshold = self._coerce_float_parameter(
            params['erbs_high_threshold'],
            'erbs_high_threshold',
        )
        if not 0.0 < low_threshold < high_threshold <= 1.0:
            raise ValueError(
                'Erbs clearness-index thresholds must satisfy '
                '0 < low < high <= 1'
            )
        low_coefficients = params['erbs_low_coefficients']
        mid_coefficients = params['erbs_mid_coefficients']
        if not isinstance(low_coefficients, list) or len(low_coefficients) != 2:
            raise ValueError('canopy.erbs_low_coefficients must contain two coefficients')
        if not isinstance(mid_coefficients, list) or len(mid_coefficients) != 5:
            raise ValueError('canopy.erbs_mid_coefficients must contain five coefficients')
        high_fraction = self._coerce_float_parameter(
            params['erbs_high_fraction'],
            'erbs_high_fraction',
        )
        if not 0.0 <= high_fraction <= 1.0:
            raise ValueError('canopy.erbs_high_fraction must be within [0, 1]')
        return params

    def _canopy_execution_enabled(self) -> bool:
        """Return the compile-time canopy topology selection."""
        return feature_manager.is_enabled('canopy_layers_enabled')

    def _locked_canopy_execution_enabled(self) -> bool:
        """Keep runtime execution aligned with the registered metadata graph."""
        enabled = self._canopy_execution_enabled()
        initialized_enabled = self._canopy_enabled_at_initialization
        if enabled != initialized_enabled:
            raise RuntimeError(
                'Canopy enablement changed after initialization; recreate and '
                'register RadiationFunctor so its IO metadata remains exact'
            )
        return enabled

    def _split_direct_diffuse(
        self,
        shortwave_down: torch.Tensor,
        extraterrestrial_horizontal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Split global shortwave with the published Erbs hourly correlation.

        Mathematical Expression:
            k_T = SW_down / SW_extraterrestrial_horizontal
            PAR_total = par_fraction_of_sw * SW_down
            f_dif = c0 + c1*k_T, k_T <= 0.22
            f_dif = sum(c_i*k_T^i), 0.22 < k_T <= 0.80
            f_dif = 0.165, k_T > 0.80
            PAR_dir = (1 - f_dif) * PAR_total
            PAR_dif = f_dif * PAR_total
            PPFD_dir = ppfd_conversion * PAR_dir
            PPFD_dif = ppfd_conversion * PAR_dif

        References:
            - Erbs et al. (1982). Solar Energy 28, 293-302
            - Jacovides et al. (2003). PAR fraction of global radiation

        Parameter meanings:
            k_T is the hourly clearness index. The thresholds and polynomial
            coefficients are ParameterManager-backed Erbs values.
            par_fraction_of_sw and ppfd_conversion are empirical spectral
            conversion factors.

        Units and route:
            Energy fluxes are W m-2, PPFD is umol photons m-2 s-1, and all
            coefficients are dimensionless. This open radiative boundary split
            neither stores nor discards energy.

        Parameters:
            shortwave_down : torch.Tensor
                Surface global horizontal shortwave radiation [W m-2]
            extraterrestrial_horizontal : torch.Tensor
                Extraterrestrial horizontal shortwave radiation [W m-2]

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - par_total: Total PAR [W m-2]
                - par_dir: Direct PAR [W m-2]
                - par_dif: Diffuse PAR [W m-2]
                - ppfd_dir: Direct PPFD [mumol m-2 s-1]
                - ppfd_dif: Diffuse PPFD [mumol m-2 s-1]
                - f_dif: Diffuse fraction [0-1]
        """
        if not isinstance(shortwave_down, torch.Tensor):
            raise TypeError('shortwave_down must be a torch.Tensor')
        if not torch.isfinite(shortwave_down).all():
            raise ValueError('shortwave_down contains non-finite values')
        if torch.any(shortwave_down < 0.0):
            raise ValueError('shortwave_down must be non-negative')
        if not isinstance(extraterrestrial_horizontal, torch.Tensor):
            raise TypeError('extraterrestrial_horizontal must be a torch.Tensor')
        extraterrestrial_horizontal = extraterrestrial_horizontal.to(
            device=shortwave_down.device,
            dtype=shortwave_down.dtype,
        )
        if not torch.isfinite(extraterrestrial_horizontal).all():
            raise ValueError('extraterrestrial_horizontal contains non-finite values')
        if torch.any(extraterrestrial_horizontal < 0.0):
            raise ValueError('extraterrestrial_horizontal must be non-negative')
        try:
            shortwave_down, extraterrestrial_horizontal = torch.broadcast_tensors(
                shortwave_down,
                extraterrestrial_horizontal,
            )
        except RuntimeError as exc:
            raise ValueError(
                'shortwave_down and extraterrestrial_horizontal are not broadcast-compatible'
            ) from exc
        if torch.any(shortwave_down > extraterrestrial_horizontal):
            raise ValueError(
                'shortwave_down exceeds extraterrestrial horizontal radiation'
            )

        par_fraction = self._coerce_parameter_tensor(
            self._get_canopy_parameter('par_fraction_of_sw'),
            shortwave_down,
            'par_fraction_of_sw',
        )
        ppfd_conversion = self._coerce_parameter_tensor(
            self._get_canopy_parameter('ppfd_conversion'),
            shortwave_down,
            'ppfd_conversion',
        )
        low_threshold = self._coerce_parameter_tensor(
            self._get_canopy_parameter('erbs_low_threshold'),
            shortwave_down,
            'erbs_low_threshold',
        )
        high_threshold = self._coerce_parameter_tensor(
            self._get_canopy_parameter('erbs_high_threshold'),
            shortwave_down,
            'erbs_high_threshold',
        )
        low_coefficients = self._get_canopy_parameter('erbs_low_coefficients')
        mid_coefficients = self._get_canopy_parameter('erbs_mid_coefficients')
        high_fraction = self._coerce_parameter_tensor(
            self._get_canopy_parameter('erbs_high_fraction'),
            shortwave_down,
            'erbs_high_fraction',
        )

        positive_extraterrestrial = extraterrestrial_horizontal > 0.0
        safe_extraterrestrial = torch.where(
            positive_extraterrestrial,
            extraterrestrial_horizontal,
            torch.ones_like(extraterrestrial_horizontal),
        )
        clearness_index = torch.where(
            positive_extraterrestrial,
            shortwave_down / safe_extraterrestrial,
            torch.zeros_like(shortwave_down),
        )

        def _evaluate_polynomial(coefficients: List[Any]) -> torch.Tensor:
            result = torch.zeros_like(clearness_index)
            for index, coefficient in enumerate(coefficients):
                coefficient_tensor = self._coerce_parameter_tensor(
                    coefficient,
                    clearness_index,
                    f'erbs_polynomial_coefficient_{index}',
                )
                result = result + coefficient_tensor * torch.pow(
                    clearness_index,
                    index,
                )
            return result

        low_diffuse = _evaluate_polynomial(low_coefficients)
        mid_diffuse = _evaluate_polynomial(mid_coefficients)
        f_dif = torch.where(
            clearness_index <= low_threshold,
            low_diffuse,
            torch.where(
                clearness_index <= high_threshold,
                mid_diffuse,
                high_fraction.expand_as(clearness_index),
            ),
        )
        if torch.any((f_dif < 0.0) | (f_dif > 1.0)):
            raise RuntimeError('Erbs diffuse fraction is outside [0, 1]')
        par_total = par_fraction * shortwave_down
        par_dif = f_dif * par_total
        par_dir = (1.0 - f_dif) * par_total
        ppfd_dir = ppfd_conversion * par_dir
        ppfd_dif = ppfd_conversion * par_dif
        par_total_check = par_dir + par_dif
        if not torch.allclose(par_total, par_total_check, rtol=1e-06):
            raise RuntimeError('PAR conservation check failed in direct-diffuse split')
        return {'par_total': par_total, 'par_dir': par_dir, 'par_dif': par_dif, 'ppfd_dir': ppfd_dir, 'ppfd_dif': ppfd_dif, 'f_dif': f_dif}

    @staticmethod
    def _coerce_float_parameter(value: Any, param_name: str) -> float:
        """Convert scalar-like parameter values to float."""
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Parameter '{param_name}' must be scalar.")
            return float(value.detach().reshape(-1)[0].tolist())
        if isinstance(value, (int, float, bool)):
            return float(value)
        if isinstance(value, str):
            return float(value)
        raise TypeError(f"Parameter '{param_name}' is not float-coercible: {type(value)!r}")

    @staticmethod
    def _coerce_parameter_tensor(
        value: Any,
        reference: torch.Tensor,
        param_name: str,
    ) -> torch.Tensor:
        """Convert a scalar parameter without detaching its autograd lineage."""
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Parameter '{param_name}' must be scalar.")
            return value.to(device=reference.device, dtype=reference.dtype).reshape(())
        try:
            return torch.tensor(
                float(value),
                device=reference.device,
                dtype=reference.dtype,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Parameter '{param_name}' is not tensor-coercible: {type(value)!r}"
            ) from exc

    @staticmethod
    def _coerce_int_parameter(value: Any, param_name: str) -> int:
        """Convert scalar-like parameter values to int."""
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Parameter '{param_name}' must be scalar.")
            return int(value.detach().reshape(-1)[0].tolist())
        if isinstance(value, (int, bool)):
            return int(value)
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            return int(value)
        raise TypeError(f"Parameter '{param_name}' is not int-coercible: {type(value)!r}")

    def _get_canopy_parameter(self, param_name: str) -> Any:
        """
        Helper method to get canopy-specific parameters.
        
        This method provides access to parameters nested under the 'canopy'
        configuration section, with proper type handling.
        
        Parameters:
            param_name : str
                Parameter name (without 'canopy.' prefix)
        Returns:
            Union[float, int, str, bool, List[Any]]: Parameter value
        """
        if param_name in self._canopy_params:
            return self._canopy_params[param_name]
        raise KeyError(f"Canopy parameter '{param_name}' not loaded via ParameterManager.")

    def _extract_parameter_value(self, param_name: str) -> Any:
        """
        Extract parameter value with default to manager and config sources.
        
        This method uses the enhanced parameter management system to get values
        from manager or config sources in priority order.
        
        Parameters:
            param_name : str
                Parameter name to retrieve
        Returns:
            Union[float, int, bool]: Extracted parameter value
        """
        if param_name in self._radiation_params:
            return self._radiation_params[param_name]
        raise KeyError(f"Radiation parameter '{param_name}' not loaded via ParameterManager.")

    def _lai_profile(self, lai_total: torch.Tensor, n_layers: int) -> torch.Tensor:
        """
        Generate LAI vertical profile distribution for canopy layers.
        
        Creates LAI distribution across canopy layers using either uniform
        distribution or Beta distribution based on configuration. The Beta
        distribution allows for more realistic canopy structure with varying
        leaf density at different heights.
        
        Mathematical Expression:
            Uniform: L_i = LAI_total / N for all layers
            Beta midpoint: x_i = (i + 0.5) / N
            log(w_i) = (a - 1) log(x_i) + (b - 1) log(1 - x_i)
            L_i = softmax(log(w))_i * LAI_total
            Conservation: Sigma L_i = LAI_total
            
        References:
            - Campbell & Norman (1998). Environmental Biophysics
            - Bonan (2019). Climate Change and Terrestrial Ecosystem Modeling
            - Song et al. (2021). JAMES 13, e2020MS002259
        
        Parameters:
            lai_total : torch.Tensor
                Total leaf area index [m^2 m^-2]
            n_layers : int
                Number of canopy layers
                
        Returns:
            torch.Tensor: LAI values for each layer from top to bottom
                Shape: [n_layers]
                Units: [m^2 m^-2]
        """
        lai_total_tensor = torch.as_tensor(lai_total)
        if lai_total_tensor.numel() != 1:
            raise ValueError('lai_total must be scalar for the grid-cell canopy model')
        if n_layers < 1:
            raise ValueError('n_layers must be positive')
        lai_total_scalar = lai_total_tensor.reshape(())
        lai_total_scalar = lai_total_scalar.to(dtype=lai_total_tensor.dtype, device=lai_total_tensor.device)

        profile_type = self._get_canopy_parameter('lai_profile')
        if profile_type == 'uniform':
            layer_lai = torch.ones(
                (n_layers,),
                dtype=lai_total_scalar.dtype,
                device=lai_total_scalar.device,
            ) * (lai_total_scalar / n_layers)
        elif profile_type == 'beta':
            beta_params = self._get_canopy_parameter('beta_params')
            if not isinstance(beta_params, list) or len(beta_params) != 2:
                raise ValueError('canopy.beta_params must contain exactly two coefficients')
            a, b = beta_params
            positions = (
                torch.arange(
                    n_layers,
                    dtype=lai_total_scalar.dtype,
                    device=lai_total_scalar.device,
                )
                + 0.5
            ) / n_layers
            a_tensor = torch.as_tensor(
                a,
                dtype=lai_total_scalar.dtype,
                device=lai_total_scalar.device,
            )
            b_tensor = torch.as_tensor(
                b,
                dtype=lai_total_scalar.dtype,
                device=lai_total_scalar.device,
            )
            log_weights = (
                (a_tensor - 1.0) * torch.log(positions)
                + (b_tensor - 1.0) * torch.log1p(-positions)
            )
            weights = torch.softmax(log_weights, dim=0)
            layer_lai = weights * lai_total_scalar
        else:
            raise ValueError(f"Unsupported canopy LAI profile type '{profile_type}'")
        actual_total = torch.sum(layer_lai)
        if not torch.allclose(actual_total, lai_total_scalar, rtol=1e-05):
            layer_lai = layer_lai * (lai_total_scalar / actual_total)
        layer_lai = torch.clamp(layer_lai, min=0.0)
        return layer_lai

    def _beer_lambert_layers(self, par_dir: torch.Tensor, par_dif: torch.Tensor, lai_total: torch.Tensor, n_layers: int) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-layer radiation transfer using Beer-Lambert law.
        
        Implements the Beer-Lambert exponential attenuation model for canopy
        radiation transfer, computing absorbed PAR for each canopy layer.
        Uses separate extinction coefficients for direct and diffuse radiation.
        
        Mathematical Expression:
            Cumulative LAI: L_cum(i) = Sigma(j=1 to i) L_j
            Direct layer absorption: APAR_dir_i = PAR_dir * [exp(-k_b*L_cum(i-1)) - exp(-k_b*L_cum(i))]
            Diffuse layer absorption: APAR_dif_i = PAR_dif * [exp(-k_d*L_cum(i-1)) - exp(-k_d*L_cum(i))]
            Total layer absorption: APAR_layer_i = APAR_dir_i + APAR_dif_i
            Fraction absorbed: fPAR = Sigma APAR_layer_i / PAR_total
            
        Conservation check: Sigma APAR_layer_i <= PAR_total (with numerical tolerance)
            
        References:
            - Monsi, M. & Saeki, T. (1953). Light factor and leaf arrangement
            - Charles-Edwards, D.A. (1982). Physiological determinants of crop growth
            - Campbell & Norman (1998). Environmental Biophysics, Ch. 15
        
        Parameters:
            par_dir : torch.Tensor
                Direct PAR at canopy top [W m-2]
            par_dif : torch.Tensor
                Diffuse PAR at canopy top [W m-2]
            lai_total : torch.Tensor
                Total leaf area index [m^2 m^-2]
            n_layers : int
                Number of canopy layers
                
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - apar_layer: Absorbed PAR by layer [W m-2], shape [n_layers]
                - fpar: Fraction of PAR absorbed [0-1]
                - transmission_dir: Direct PAR transmission profile [W m-2]
                - transmission_dif: Diffuse PAR transmission profile [W m-2]
        """
        k_b = self._coerce_parameter_tensor(
            self._get_canopy_parameter('k_b'),
            par_dir,
            'k_b',
        )
        k_d = self._coerce_parameter_tensor(
            self._get_canopy_parameter('k_d'),
            par_dif,
            'k_d',
        )
        if torch.any(k_b <= 0) or torch.any(k_d <= 0):
            raise ValueError(f'Extinction coefficients must be positive: k_b={k_b}, k_d={k_d}')
        if torch.all(lai_total <= 1e-06):
            apar_layer = torch.zeros(n_layers, dtype=par_dir.dtype, device=par_dir.device)
            fpar = torch.zeros_like(par_dir)
            transmission_dir = par_dir.clone()
            transmission_dif = par_dif.clone()
            return {'apar_layer': apar_layer, 'fpar': fpar, 'transmission_dir': transmission_dir, 'transmission_dif': transmission_dif}
        layer_lai = self._lai_profile(lai_total, n_layers)
        lai_cumulative = torch.cumsum(layer_lai, dim=0)
        lai_cumulative_prev = torch.cat([torch.zeros(1, dtype=layer_lai.dtype, device=layer_lai.device), lai_cumulative[:-1]])
        transmission_dir_top = torch.exp(-torch.clamp(k_b * lai_cumulative_prev, max=20.0))
        transmission_dir_bottom = torch.exp(-torch.clamp(k_b * lai_cumulative, max=20.0))
        transmission_dif_top = torch.exp(-torch.clamp(k_d * lai_cumulative_prev, max=20.0))
        transmission_dif_bottom = torch.exp(-torch.clamp(k_d * lai_cumulative, max=20.0))
        apar_dir_layer = par_dir * (transmission_dir_top - transmission_dir_bottom)
        apar_dif_layer = par_dif * (transmission_dif_top - transmission_dif_bottom)
        apar_layer = apar_dir_layer + apar_dif_layer
        apar_layer = torch.clamp(apar_layer, min=0.0)
        par_total = par_dir + par_dif
        apar_total = torch.sum(apar_layer)
        if par_total > 1e-06:
            fpar = apar_total / par_total
        else:
            fpar = torch.zeros_like(par_total)
        fpar = torch.clamp(fpar, min=0.0, max=1.0)
        transmission_dir = par_dir * transmission_dir_bottom[-1]
        transmission_dif = par_dif * transmission_dif_bottom[-1]
        if not torch.allclose(
            apar_total + transmission_dir + transmission_dif,
            par_total,
            rtol=1e-5,
            atol=1e-7,
        ):
            raise RuntimeError('Beer-Lambert canopy energy balance failed')
        return {'apar_layer': apar_layer, 'fpar': fpar, 'transmission_dir': transmission_dir, 'transmission_dif': transmission_dif}

    def _compute_canopy_radiation(
        self,
        shortwave_down: torch.Tensor,
        extraterrestrial_horizontal: torch.Tensor,
        lai_total: torch.Tensor,
        return_diagnostics: bool = False,
        radiation_components: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Main entry point for canopy multi-layer radiation calculations.
        
        This method orchestrates the complete canopy radiation transfer calculation,
        from direct-diffuse separation through multi-layer absorption. It serves
        as the primary interface between the main radiation model and the canopy
        sub-model, ensuring consistent data flow and error handling.
        
        Process Flow:
            1. Direct/diffuse separation of shortwave radiation
            2. Internal PAR partitioning
            3. LAI profile generation (if LAI provided)
            4. Multi-layer Beer-Lambert radiation transfer
            5. Output packaging and diagnostics
            
        Parameters:
            shortwave_down : torch.Tensor
                Incoming shortwave radiation [W m-2]
            extraterrestrial_horizontal : torch.Tensor
                Extraterrestrial horizontal shortwave radiation [W m-2]
            lai_total : torch.Tensor
                Total leaf area index [m^2 m^-2]
            return_diagnostics : bool
                Whether to include detailed diagnostic outputs
                
        Returns:
            Dict[str, torch.Tensor]: Canopy radiation outputs:
                Basic outputs (always included):
                - fpar: Fraction of PAR absorbed [0-1]
                - apar_layer: Absorbed PAR by layer [W m-2]
                
                Diagnostic outputs (if return_diagnostics=True):
                - diagnostics.radiation.par_energy: Total PAR energy [W m-2]
                - diagnostics.radiation.par_direct: Direct PAR [W m-2]
                - diagnostics.radiation.par_diffuse: Diffuse PAR [W m-2]
                - diagnostics.radiation.ppfd_direct: Direct PPFD [umol photons m-2 s-1]
                - diagnostics.radiation.ppfd_diffuse: Diffuse PPFD [umol photons m-2 s-1]
                - diagnostics.radiation.diffuse_fraction: Diffuse fraction [0-1]
                - diagnostics.radiation.transmission_direct: Direct transmission [W m-2]
                - diagnostics.radiation.transmission_diffuse: Diffuse transmission [W m-2]
        """
        if not self._locked_canopy_execution_enabled():
            raise RuntimeError('Canopy radiation is not enabled')
        if not isinstance(shortwave_down, torch.Tensor):
            raise TypeError('shortwave_down must be a torch.Tensor')
        if not torch.isfinite(shortwave_down).all():
            raise ValueError('shortwave_down contains non-finite values')
        if torch.any(shortwave_down < 0.0):
            raise ValueError('shortwave_down must be non-negative')
        if shortwave_down.numel() != 1:
            raise ValueError('shortwave_down must be scalar for the grid-cell canopy model')
        if lai_total is None:
            raise KeyError('leaf_area_index is required when canopy radiation is enabled')
        if not isinstance(lai_total, torch.Tensor):
            raise TypeError('leaf_area_index must be a torch.Tensor')
        lai_total = lai_total.to(device=shortwave_down.device, dtype=shortwave_down.dtype)
        if lai_total.numel() != 1:
            raise ValueError('leaf_area_index must be scalar for the grid-cell canopy model')
        if not torch.isfinite(lai_total).all():
            raise ValueError('leaf_area_index contains non-finite values')
        if torch.any(lai_total < 0.0):
            raise ValueError('leaf_area_index must be non-negative')
        n_layers = self._coerce_int_parameter(self._get_canopy_parameter('layers'), 'layers')
        if radiation_components is None:
            radiation_components = self._split_direct_diffuse(
                shortwave_down,
                extraterrestrial_horizontal,
            )
        par_dir = radiation_components['par_dir']
        par_dif = radiation_components['par_dif']
        layer_results = self._beer_lambert_layers(
            par_dir,
            par_dif,
            lai_total,
            n_layers,
        )
        outputs = {
            'fpar': layer_results['fpar'],
            'apar_layer': layer_results['apar_layer'],
        }
        if return_diagnostics:
            outputs.update({
                'diagnostics.radiation.par_energy': radiation_components['par_total'],
                'diagnostics.radiation.par_direct': radiation_components['par_dir'],
                'diagnostics.radiation.par_diffuse': radiation_components['par_dif'],
                'diagnostics.radiation.ppfd_direct': radiation_components['ppfd_dir'],
                'diagnostics.radiation.ppfd_diffuse': radiation_components['ppfd_dif'],
                'diagnostics.radiation.diffuse_fraction': radiation_components['f_dif'],
                'diagnostics.radiation.transmission_direct': layer_results['transmission_dir'],
                'diagnostics.radiation.transmission_diffuse': layer_results['transmission_dif'],
            })
        for key, value in outputs.items():
            if not torch.is_tensor(value):
                raise TypeError(f"Canopy output '{key}' must be a torch.Tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"Canopy output '{key}' contains non-finite values")
        return outputs

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Check the radiative energy identity instead of summing flux components."""
        _ = input_state
        required = ('net_radiation', 'shortwave_net', 'longwave_net')
        missing = [key for key in required if key not in output_updates]
        if missing:
            raise KeyError(
                'Radiation energy audit requires outputs: ' + ', '.join(missing)
            )

        def _output_tensor(key: str) -> torch.Tensor:
            value = output_updates[key]
            if hasattr(value, 'value'):
                value = value.value
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Radiation conservation output '{key}' must be a tensor")
            return value

        # References: Allen et al. (1998); Monteith and Unsworth (2013).
        # Mathematical expression: residual = R_net - SW_net - LW_net = 0.
        # Parameter meanings: R_net is total net radiation; SW_net and LW_net
        # are the net shortwave and longwave boundary energy fluxes [W m-2].
        residual = (
            _output_tensor('net_radiation')
            - _output_tensor('shortwave_net')
            - _output_tensor('longwave_net')
        )
        if not torch.isfinite(residual).all():
            raise ValueError('Radiation energy conservation residual is non-finite')
        magnitude = _scalar_from_tensor(torch.max(torch.abs(residual)))
        return {'energy': magnitude}

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare the typed open-boundary flux used by the level-2 wrapper.

        References:
          - Allen et al. (1998), FAO-56 radiation balance.
          - van der Schaft and Jeltsema (2014), port-Hamiltonian systems.
        Mathematical expression:
          R_net = SW_net + LW_net.
        Parameter meanings:
          R_net is an instantaneous boundary energy flux [W m-2], not a
          stored-energy state or a Hamiltonian effort.
        """

        return (
            CochainSpaceSpec(
                complex_id="atmosphere_radiation_balance",
                degree=0,
                primality="primal",
                n_dofs=1,
                units="W m-2",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="radiative_boundary_flux",
                orientation_convention="positive_into_surface",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        radiation_keys = tuple(
            f"atmosphere.radiation.{key}"
            for key in sorted(self._radiation_params.keys())
        )
        canopy_keys = tuple(
            f"atmosphere.radiation.canopy.{key}"
            for key, _, _ in sorted(
                self._CANOPY_SHARED_PARAMETER_SPECS
                + self._CANOPY_LAYER_PARAMETER_SPECS
            )
        )
        return radiation_keys + canopy_keys

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Return no Hodge term because no storage-to-flux map is declared."""

        _ = complex_bundle, resolved_parameters
        return ()

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no Hamiltonian term for an instantaneous boundary flux."""

        _ = state
        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return no entropy production without a thermodynamic force law."""

        _ = state
        return {}

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible structure for the level-2 boundary wrapper."""

        _ = state, complex_bundle
        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no Onsager operator without a validated constitutive map."""

        _ = state, complex_bundle
        return None

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Return no power port without a conjugate effort-flow pair."""

        return ()

    def compute(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
    ) -> Mapping[str, Any]:
        """Run the canonical calculation through the existing Functor lifecycle."""

        return self.forward(dict(state), time, dt)

    def _calculate_actual_vapor_pressure(
        self,
        air_temp: torch.Tensor,
        relative_humidity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate actual vapor pressure without evaluating or owning VPD.
        
        Mathematical Expression:
            e_sat = e_0 * exp(a * T_c / (T_c + b))
            e_a = RH * e_sat

        References:
            - Tetens (1930), saturation vapor pressure formulation
            - Allen et al. (1998), FAO-56, equations 11 and 19
            - Brutsaert (1975), clear-sky emissivity using actual vapor pressure
        
        Parameters:
            air_temp : torch.Tensor
                Air temperature [K]
            relative_humidity : torch.Tensor
                Relative humidity [dimensionless 0-1]
                
        Returns:
            torch.Tensor: Actual vapor pressure [Pa]
        """
        # References: Tetens (1930); Allen et al. (1998); Brutsaert (1975).
        # Mathematical expression: e_s = e_0 exp(a T_c / (T_c + b)); e_a = RH e_s.
        # Parameter meanings: e_0 is saturation pressure [Pa], a is dimensionless,
        # b is the temperature offset [degC], RH is relative humidity [1], and
        # e_a is the actual vapor pressure [Pa] used only by the longwave law.
        # Open system route: e_a is a read-only atmospheric observation; this
        # process neither calculates nor writes vapor pressure deficit.
        saturation_vapor_pressure = compute_saturation_vapor_pressure(
            air_temp,
            tetens_base_pressure=float(self._tetens_base_pressure),
            tetens_coeff_a=float(self._tetens_coefficient_a),
            tetens_coeff_b=float(self._tetens_coefficient_b),
            kelvin_offset=float(self._standard_temperature),
        )
        vapor_pressure = relative_humidity * saturation_vapor_pressure
        if not torch.isfinite(vapor_pressure).all():
            raise ValueError('calculated vapor_pressure contains non-finite values')
        return vapor_pressure

    def _calculate_shortwave_radiation(self, cos_zenith: torch.Tensor, sw_potential: torch.Tensor, cloud_fraction: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Calculate incoming shortwave radiation with atmospheric and cloud effects.
        
        Mathematical Expression:
            SW_clear = SW_potential * tau_atm * max(0, cos(theta))
            SW_down = SW_clear * (1 - f_cloud * C_cloud)
            
        References:
            - Campbell & Norman (1998). Environmental Biophysics, Chapter 11
            - FAO-56 equation for clear-sky radiation
        
        Parameters:
            cos_zenith : torch.Tensor
                Cosine of solar zenith angle [dimensionless]
            sw_potential : torch.Tensor
                Extraterrestrial shortwave radiation [W m-2]
            cloud_fraction : Optional[torch.Tensor]
                Cloud fraction [dimensionless 0-1]
                
        Returns:
            torch.Tensor: Incoming shortwave radiation [W m-2]
        """
        tau_atm = self._extract_parameter_value('atmospheric_transmittance')
        cloud_factor = self._extract_parameter_value('cloud_attenuation_factor')
        cos_zenith_positive = torch.clamp(cos_zenith, min=0.0)
        sw_clear = sw_potential * tau_atm * cos_zenith_positive
        if cloud_fraction is not None:
            cloud_reduction = cloud_fraction * cloud_factor
            sw_unclamped = sw_clear * (1.0 - cloud_reduction)
        else:
            sw_unclamped = sw_clear
        # References: Campbell and Norman (1998), clear-sky and cloud shortwave balance.
        # Mathematical expression: SW_down = max(0, SW_pot * tau_atm * cos(theta) * f_cloud).
        # Parameter meanings: tau_atm and cloud_factor are ParameterManager-backed optical coefficients.
        sw_down = torch.clamp(sw_unclamped, min=0.0)
        self._last_shortwave_clipping_residual = sw_down - sw_unclamped
        return sw_down

    def _calculate_longwave_radiation(self, air_temp: torch.Tensor, surface_temp: torch.Tensor, vapor_pressure: torch.Tensor, surface_emissivity: torch.Tensor, cloud_fraction: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """
        Calculate longwave radiation components using empirical atmospheric emissivity.
        
        Mathematical Expression:
            epsilon_clear = C1 * (e_a / T_air)^C2  # Brutsaert (1975) formula
            epsilon_atm = epsilon_clear * (1-fc) + epsilon_cloud * fc  # Cloud enhancement
            LW_down = epsilon_atm * sigma * T_air^4   # Atmospheric emission [W m-2]
            LW_emitted = epsilon_sfc * sigma * T_sfc^4  # Surface emission [W m-2]
            LW_reflected = (1 - epsilon_sfc) * LW_down  # Reflection [W m-2]
            LW_up = LW_emitted + LW_reflected  # Total outgoing longwave [W m-2]
            LW_net = LW_down - LW_up  # Absorbed incident minus emitted [W m-2]
            
        References:
            - Brutsaert, W. (1975). On a derivable formula for long-wave radiation
            - Campbell & Norman (1998). Environmental Biophysics, Chapter 10
        
        Parameters:
            air_temp : torch.Tensor
                Air temperature [K]
            surface_temp : torch.Tensor
                Surface temperature [K]
            vapor_pressure : torch.Tensor
                Actual vapor pressure [Pa]
            surface_emissivity : torch.Tensor
                Surface emissivity [dimensionless]
            cloud_fraction : Optional[torch.Tensor]
                Cloud fraction [dimensionless 0-1] for enhanced longwave emission
                
        Returns:
            Dict[str, torch.Tensor]: Longwave radiation components
        """
        brutsaert_coef = self._extract_parameter_value('brutsaert_coefficient')
        brutsaert_exp = self._extract_parameter_value('brutsaert_exponent')
        emissivity_clear_min = self._extract_parameter_value('emissivity_clear_min')
        emissivity_clear_max = self._extract_parameter_value('emissivity_clear_max')
        cloud_emissivity = self._extract_parameter_value('cloud_emissivity')
        pa_to_hpa = get_physical_constant(
            'relic.physical.constants.conversions.pa_to_hpa'
        )
        pa_to_hpa = torch.as_tensor(pa_to_hpa, device=air_temp.device, dtype=air_temp.dtype)
        vapor_pressure_hpa = vapor_pressure * pa_to_hpa
        epsilon_raw = brutsaert_coef * torch.pow(vapor_pressure_hpa / air_temp, brutsaert_exp)
        # References: Brutsaert (1975), atmospheric longwave emissivity.
        # Mathematical expression: epsilon_clear = clamp(C1 * (e_a / T_air)^C2, epsilon_min, epsilon_max).
        # Parameter meanings: C1, C2, epsilon_min, and epsilon_max are ParameterManager-backed bounds.
        epsilon_clear = torch.clamp(epsilon_raw, min=emissivity_clear_min, max=emissivity_clear_max)
        self._last_emissivity_clipping_residual = epsilon_clear - epsilon_raw
        if cloud_fraction is not None:
            epsilon_cloudy = torch.ones_like(epsilon_clear) * cloud_emissivity
            epsilon_atm = epsilon_clear * (1.0 - cloud_fraction) + epsilon_cloudy * cloud_fraction
        else:
            epsilon_atm = epsilon_clear
        # References:
        #   Brutsaert (1975), doi:10.1029/WR011i005p00742, incident atmospheric LW;
        #   Nicodemus (1965), doi:10.1364/AO.4.000767, reciprocal opaque surfaces;
        #   Noah-MP Technical Note, Eqs. 3.6.330 and 3.6.416-417.
        # Mathematical expression:
        #   L_down=eps_atm*sigma*T_air^4; L_emitted=eps_sfc*sigma*T_sfc^4;
        #   L_reflected=(1-eps_sfc)*L_down; L_up=L_emitted+L_reflected;
        #   L_net=L_down-L_up=eps_sfc*L_down-L_emitted.
        # Parameter meanings:
        #   eps_atm is the existing Brutsaert/cloud atmospheric emissivity;
        #   eps_sfc is supplied surface LW emissivity/absorptivity; sigma is the
        #   SSOT Stefan-Boltzmann constant; T_air/T_sfc are supplied temperatures;
        #   L_down is incident LW and L_up is total outgoing LW.
        # Units:
        #   fluxes [W m-2]; sigma [W m-2 K-4]; temperatures [K]; emissivities [1].
        # Domain/constraints:
        #   reciprocal opaque diffuse-grey surface with matched broadband LW
        #   absorptivity/emissivity; finite tensors and existing temperature/
        #   unity bounds validated at ingress; no spectral-material generalization.
        # Sign convention:
        #   L_down and signed L_net are inward-positive; emitted/reflected/L_up
        #   are outward-positive magnitudes; negative L_net is cooling.
        lw_down = epsilon_atm * self._stefan_boltzmann * torch.pow(air_temp, 4)
        lw_emitted = surface_emissivity * self._stefan_boltzmann * torch.pow(surface_temp, 4)
        lw_reflected = (1.0 - surface_emissivity) * lw_down
        lw_up = lw_emitted + lw_reflected
        lw_net = lw_down - lw_up
        return {
            'longwave_down': lw_down,
            'longwave_up': lw_up,
            'longwave_emitted': lw_emitted,
            'longwave_reflected': lw_reflected,
            'longwave_net': lw_net,
            'atmospheric_emissivity': epsilon_atm,
        }

    @staticmethod
    def _require_state_tensor(
        state: StateDict,
        key: str,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return one required finite floating-point state cochain."""
        if key not in state:
            raise KeyError(f"Missing required radiation state '{key}'")
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Radiation state '{key}' must be a torch.Tensor")
        if not torch.is_floating_point(value):
            raise TypeError(f"Radiation state '{key}' must use a floating-point dtype")
        if value.numel() == 0:
            raise ValueError(f"Radiation state '{key}' must not be empty")
        if reference is not None:
            value = value.to(device=reference.device, dtype=reference.dtype)
        if not torch.isfinite(value).all():
            raise ValueError(f"Radiation state '{key}' contains non-finite values")
        return value

    def _resolve_forward_inputs(self, state: StateDict) -> _ResolvedRadiationInputs:
        air_temp = self._require_state_tensor(state, 'air_temperature')
        try:
            validate_temperature_range(air_temp, *RADIATION_AIR_TEMPERATURE_DOMAIN_K)
        except ValueError as exc:
            raise ValueError(f'air_temperature is invalid: {exc}') from exc

        sw_potential = self._require_state_tensor(
            state,
            'incoming_shortwave_potential',
            air_temp,
        )
        if torch.any(sw_potential < 0.0):
            raise ValueError('incoming_shortwave_potential must be non-negative')

        cos_zenith = self._require_state_tensor(state, 'cos_zenith', air_temp)
        if torch.any((cos_zenith < -1.0) | (cos_zenith > 1.0)):
            raise ValueError('cos_zenith must be within [-1, 1]')

        surface_temp = self._require_state_tensor(state, 'surface_temperature', air_temp)
        try:
            validate_temperature_range(surface_temp, *RADIATION_SURFACE_TEMPERATURE_DOMAIN_K)
        except ValueError as exc:
            raise ValueError(f'surface_temperature is invalid: {exc}') from exc

        relative_humidity = self._require_state_tensor(
            state,
            'relative_humidity',
            air_temp,
        )
        surface_albedo = self._require_state_tensor(state, 'surface_albedo', air_temp)
        surface_emissivity = self._require_state_tensor(
            state,
            'surface_emissivity',
            air_temp,
        )
        cloud_fraction = self._require_state_tensor(state, 'cloud_fraction', air_temp)
        validate_unity_range(relative_humidity, 'relative_humidity')
        validate_unity_range(surface_albedo, 'surface_albedo')
        validate_unity_range(surface_emissivity, 'surface_emissivity')
        validate_unity_range(cloud_fraction, 'cloud_fraction')
        try:
            torch.broadcast_shapes(
                air_temp.shape,
                sw_potential.shape,
                cos_zenith.shape,
                surface_temp.shape,
                relative_humidity.shape,
                surface_albedo.shape,
                surface_emissivity.shape,
                cloud_fraction.shape,
            )
        except RuntimeError as exc:
            raise ValueError('Radiation state shapes are not broadcast-compatible') from exc
        vapor_pressure = self._calculate_actual_vapor_pressure(
            air_temp,
            relative_humidity,
        )
        return {
            'air_temp': air_temp,
            'sw_potential': sw_potential,
            'cos_zenith': cos_zenith,
            'surface_temp': surface_temp,
            'vapor_pressure': vapor_pressure,
            'surface_albedo': surface_albedo,
            'surface_emissivity': surface_emissivity,
            'cloud_fraction': cloud_fraction,
        }

    def _compute_canopy_outputs(
        self,
        state: StateDict,
        shortwave_down: torch.Tensor,
        extraterrestrial_horizontal: torch.Tensor,
        return_diagnostics: bool,
        radiation_components: Dict[str, torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if not self._locked_canopy_execution_enabled():
            return None
        if 'leaf_area_index' not in state:
            raise KeyError('leaf_area_index is required when canopy radiation is enabled')
        lai_total = state['leaf_area_index']
        if not isinstance(lai_total, torch.Tensor):
            raise TypeError('leaf_area_index must be a torch.Tensor')
        if not torch.isfinite(lai_total).all():
            raise ValueError('leaf_area_index contains non-finite values')
        return self._compute_canopy_radiation(
            shortwave_down=shortwave_down,
            extraterrestrial_horizontal=extraterrestrial_horizontal,
            lai_total=lai_total,
            return_diagnostics=return_diagnostics,
            radiation_components=radiation_components,
        )

    def _validate_energy_conservation(
        self,
        shortwave_down: torch.Tensor,
        shortwave_net: torch.Tensor,
        net_radiation: torch.Tensor,
        longwave_components: Dict[str, torch.Tensor],
        air_temp: torch.Tensor,
        surface_temp: torch.Tensor,
        surface_albedo: torch.Tensor,
        surface_emissivity: torch.Tensor,
        cos_zenith: torch.Tensor,
        canopy_results: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, Any]]:
        validator = self._get_conservation_validator()
        if validator is None:
            return None
        canopy_validation_outputs: Optional[Dict[str, torch.Tensor]] = None
        if canopy_results is not None:
            par_fraction = self._coerce_float_parameter(
                self._get_canopy_parameter('par_fraction_of_sw'),
                'par_fraction_of_sw',
            )
            canopy_validation_outputs = {
                'par_total': shortwave_down * par_fraction,
                'fpar': canopy_results['fpar'],
                'apar_layer': canopy_results['apar_layer'],
            }
        conservation_report_obj = validator.validate_energy_conservation(
            radiation_fluxes={
                'shortwave_down': shortwave_down,
                'longwave_down': longwave_components['longwave_down'],
                'longwave_up': longwave_components['longwave_up'],
                'shortwave_net': shortwave_net,
                'longwave_net': longwave_components['longwave_net'],
                'net_radiation': net_radiation,
            },
            radiation_state={
                'air_temperature': air_temp,
                'surface_temperature': surface_temp,
                'surface_albedo': surface_albedo,
                'surface_emissivity': surface_emissivity,
                'cos_zenith': cos_zenith,
            },
            canopy_outputs=canopy_validation_outputs,
        )
        conservation_report = asdict(conservation_report_obj)
        violation_type = conservation_report.get('violation_type')
        if violation_type is not None:
            violation_value = getattr(violation_type, 'value', None)
            if violation_value is not None:
                conservation_report['violation_type'] = violation_value
        self._last_conservation_report = conservation_report
        return conservation_report

    def _append_forward_diagnostics(
        self,
        fluxes: FluxDict,
        radiation_components: Dict[str, torch.Tensor],
        longwave_components: Dict[str, torch.Tensor],
        surface_albedo: torch.Tensor,
        surface_emissivity: torch.Tensor,
        cos_zenith: torch.Tensor,
        vapor_pressure: torch.Tensor,
        conservation_report: Optional[Dict[str, Any]],
    ) -> None:
        fluxes.update({
            'diagnostics.radiation.par_energy': radiation_components['par_total'],
            'diagnostics.radiation.par_direct': radiation_components['par_dir'],
            'diagnostics.radiation.par_diffuse': radiation_components['par_dif'],
            'diagnostics.radiation.ppfd_direct': radiation_components['ppfd_dir'],
            'diagnostics.radiation.ppfd_diffuse': radiation_components['ppfd_dif'],
            'diagnostics.radiation.diffuse_fraction': radiation_components['f_dif'],
            'diagnostics.radiation.longwave_up': longwave_components['longwave_up'],
            'diagnostics.radiation.longwave_emitted': longwave_components['longwave_emitted'],
            'diagnostics.radiation.longwave_reflected': longwave_components['longwave_reflected'],
            'diagnostics.radiation.surface_albedo_used': surface_albedo,
            'diagnostics.radiation.surface_emissivity_used': surface_emissivity,
            'diagnostics.radiation.cos_zenith_used': cos_zenith,
            'diagnostics.radiation.vapor_pressure_calculated': vapor_pressure,
        })
        _ = conservation_report
        if self._last_shortwave_clipping_residual is not None:
            fluxes[
                'diagnostics.radiation.shortwave_clipping_residual'
            ] = self._last_shortwave_clipping_residual.detach().clone()
        if self._last_emissivity_clipping_residual is not None:
            fluxes[
                'diagnostics.radiation.emissivity_clipping_residual'
            ] = self._last_emissivity_clipping_residual.detach().clone()

    def _wrap_forward_outputs(
        self,
        values: Dict[str, torch.Tensor],
    ) -> FluxDict:
        """Attach instantaneous StateManager semantics and units to outputs."""
        units = {
            **self._CANONICAL_OUTPUT_UNITS,
            **self._DIAGNOSTIC_OUTPUT_UNITS,
            **self._CANOPY_OUTPUT_UNITS,
            **self._CANOPY_DIAGNOSTIC_OUTPUT_UNITS,
        }
        wrapped: FluxDict = {}
        for key, value in values.items():
            if key not in units:
                raise KeyError(f"Radiation output '{key}' has no declared unit contract")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Radiation output '{key}' must be a torch.Tensor")
            if not torch.isfinite(value).all():
                raise ValueError(f"Radiation output '{key}' contains non-finite values")
            wrapped[key] = UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                write_mode=(
                    StateWriteMode.DERIVED
                    if key in self._DIAGNOSTIC_OUTPUT_UNITS
                    or key in self._CANOPY_DIAGNOSTIC_OUTPUT_UNITS
                    else None
                ),
                unit=units[key],
                per_timestep=False,
                source_functor=self.name,
            )
        return wrapped

    def forward(
        self,
        state: StateDict,
        time: float,
        dt: Union[float, torch.Tensor],
        **kwargs: Any,
    ) -> FluxDict:
        """
        Compute atmospheric radiation fluxes for surface energy balance.
        
        This method calculates shortwave and longwave radiation components
        including atmospheric transmission, cloud effects, and surface properties.
        Provides net radiation for coupling with evapotranspiration and other
        energy balance processes.
        
        Parameters:
            state : StateDict
                Current system state containing meteorological variables
                Required keys: air_temperature, incoming_shortwave_potential,
                cos_zenith, surface_temperature, relative_humidity,
                surface_albedo, surface_emissivity, and cloud_fraction.
                leaf_area_index is additionally required when canopy radiation
                is enabled.
            time : float
                Current simulation time [s]
            dt : float
                Time step [s]
                
        Returns:
            FluxDict: Explicit REPLACE updates with the following keys and units:
                - shortwave_down: Incoming shortwave radiation [W m-2]
                - longwave_down: Downward longwave radiation [W m-2]
                - net_radiation: Net radiation (Rn = SW_net + LW_net) [W m-2]
                - shortwave_net: Net shortwave (SW_down - SW_up) [W m-2]
                - longwave_net: Net longwave (LW_down - LW_up) [W m-2]
                - atmospheric_emissivity: Effective atmospheric emissivity [dimensionless]
                - ppfd: Photosynthetic photon flux density [umol photons m-2 s-1]
                - diagnostics.radiation.*: Declared read-only radiation
                  components and explicit clipping residuals
        
        Raises:
            KeyError: If a required state variable is missing
            TypeError: If state variables are not floating-point tensors
            ValueError: If inputs, outputs, or dt are invalid
        """
        # References: Brutsaert (1975); Campbell and Norman (1998);
        # Jacovides et al. (2003); McCree (1972).
        # Mathematical expression: R_net = (1-alpha) * SW_down + LW_down - LW_up;
        # PPFD = SW_down * f_PAR * c_PPFD.
        # Parameter meanings: alpha is surface shortwave albedo; SW_down/LW_down
        # are incident fluxes; LW_up includes surface-emitted and reflected LW;
        # f_PAR is the PAR energy fraction and c_PPFD the photon conversion.
        # Units: radiative energy fluxes [W m-2], PPFD [umol photons m-2 s-1],
        # alpha/f_PAR [1], c_PPFD [umol J-1].
        # Domain/constraints: finite tensor inputs with existing physical bounds;
        # reciprocal opaque grey LW surface; one selected canopy topology.
        # Sign convention: incident and net fluxes inward-positive, total LW_up
        # outward-positive; net cooling is retained and PPFD is nonnegative.
        # Open system route: all retained values are instantaneous boundary fluxes
        # or read-only diagnostics; no prognostic energy storage is mutated here.
        unsupported_options = set(kwargs)
        if unsupported_options:
            raise TypeError(
                'Unsupported RadiationFunctor.forward options: '
                + ', '.join(sorted(unsupported_options))
            )
        if isinstance(dt, torch.Tensor):
            if dt.numel() != 1:
                raise TypeError('dt must be a scalar finite positive number')
            dt_value = _scalar_from_tensor(dt)
        elif isinstance(dt, (int, float)) and not isinstance(dt, bool):
            dt_value = float(dt)
        else:
            raise TypeError('dt must be a finite positive number')
        if not math.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError('dt must be positive and finite')
        _ = time
        self._last_shortwave_clipping_residual = None
        self._last_emissivity_clipping_residual = None
        inputs = self._resolve_forward_inputs(state)
        air_temp = inputs['air_temp']
        sw_potential = inputs['sw_potential']
        cos_zenith = inputs['cos_zenith']
        surface_temp = inputs['surface_temp']
        vapor_pressure = inputs['vapor_pressure']
        surface_albedo = inputs['surface_albedo']
        surface_emissivity = inputs['surface_emissivity']
        cloud_fraction = inputs['cloud_fraction']

        extraterrestrial_horizontal = sw_potential * torch.clamp(
            cos_zenith,
            min=0.0,
        )
        shortwave_down = self._calculate_shortwave_radiation(cos_zenith, sw_potential, cloud_fraction)
        radiation_components = self._split_direct_diffuse(
            shortwave_down,
            extraterrestrial_horizontal,
        )
        ppfd = (
            radiation_components['ppfd_dir']
            + radiation_components['ppfd_dif']
        )
        shortwave_net = (1.0 - surface_albedo) * shortwave_down
        longwave_components = self._calculate_longwave_radiation(air_temp, surface_temp, vapor_pressure, surface_emissivity, cloud_fraction)
        net_radiation = shortwave_net + longwave_components['longwave_net']
        fluxes = {
            'shortwave_down': shortwave_down,
            'longwave_down': longwave_components['longwave_down'],
            'net_radiation': net_radiation,
            'shortwave_net': shortwave_net,
            'longwave_net': longwave_components['longwave_net'],
            'atmospheric_emissivity': longwave_components['atmospheric_emissivity'],
            'ppfd': ppfd,
        }
        canopy_results = self._compute_canopy_outputs(
            state,
            shortwave_down,
            extraterrestrial_horizontal,
            True,
            radiation_components,
        )
        if canopy_results is not None:
            fluxes.update(canopy_results)
        conservation_report = self._validate_energy_conservation(
            shortwave_down,
            shortwave_net,
            net_radiation,
            longwave_components,
            air_temp,
            surface_temp,
            surface_albedo,
            surface_emissivity,
            cos_zenith,
            canopy_results,
        )
        self._append_forward_diagnostics(
            fluxes,
            radiation_components,
            longwave_components,
            surface_albedo,
            surface_emissivity,
            cos_zenith,
            vapor_pressure,
            conservation_report,
        )
        return self._wrap_forward_outputs(fluxes)

__all__ = ['RadiationFunctor', 'RadiationField']

