"""
Atmospheric Turbulence Process Module for RELIC

This module implements surface layer turbulence calculations using
Monin-Obukhov Similarity Theory (MOST) for boundary layer applications.
Provides friction velocity, aerodynamic resistance, and stability parameters
for coupling with energy balance and mass transfer processes.

Main Implementation:
    TurbulenceFunctor: Primary class implementing MOST calculations with full
                      parameter system integration and TransportMechanism interface

Physical Framework:
    MOST foundation: Monin-Obukhov length L characterizes stability
    Neutral: L -> inf, stable: L > 0, unstable: L < 0
    Wind profile: U(z) = (u*/k) * fm
    Scalar resistance: rah = fh / (k * u*)

Mathematical Expressions:
    1. fm = ln((z-d)/z0m) - psim((z-d)/L) + psim(z0m/L)
    2. fh = ln((z-d)/z0h) - psih((z-d)/L) + psih(z0h/L)
    3. Friction velocity: u* = k * U / fm
    4. Monin-Obukhov length: L = -rho * cp * T * u*^3 / (k * g * H)
    3. Stability functions (Businger et al. 1971):
       - Stable: psim = psih = -5 * z/L
       - Unstable: psim = 2*ln((1+x)/2) + ln((1+x^2)/2) - 2*atan(x) + pi/2
       - where x = (1 - 16*z/L)^(1/4)
    5. Aerodynamic resistance: rah = fh / (k * u*)
    6. Heat flux: H = rho * cp * u* * theta*
    7. Moist-air density of the inputs: rho = (p - (1 - eps) e) / (R_d * Ta),
       e = RH * e_s(Ta); rho enters H, L and tau = rho * u*^2

References:
    - Monin, A.S. & Obukhov, A.M. (1954). Basic laws of turbulent mixing
    - Businger, J.A. et al. (1971). Flux-profile relationships in the atmospheric surface layer
    - Garratt, J.R. (1992). The Atmospheric Boundary Layer
    - Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology
    - Holton, J.R. & Hakim, G.J. (2012). An Introduction to Dynamic Meteorology,
      5th ed., Eqs. 2.63-2.64 (moist-air equation of state)
    - Allen, R.G. et al. (1998). FAO-56, Eqs. 11 and 19 (e_s and e = RH * e_s)

Scientific Validation:
    - Neutral limit: psim = psih = 0, logarithmic profiles
    - Stable stratification reduces mixing (larger ra)
    - Unstable stratification enhances mixing (smaller ra)
    - Physical bounds: u* > 0 and ParameterManager-backed bounds for L

Units (UDUNITS-compatible metadata):
    - Wind speed: m s-1
    - Friction velocity: m s-1
    - Roughness lengths: m
    - Aerodynamic resistance: s m^-1
    - Monin-Obukhov length: m
    - Heat flux: W m-2
    - Stability parameter: dimensionless
    - Atmospheric pressure and vapour pressure: Pa
    - Relative humidity: 1
    - Air density: kg m-3

"""

import torch
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

# Canonical MOST operators own all scientific formulas. This Functor owns only
# lifecycle, input validation, parameter binding, and UpdateItem routing.
from .most import (
    aerodynamic_resistance as most_aerodynamic_resistance,
    iterative_most_solution,
    momentum_flux as most_momentum_flux,
    zeta as most_zeta,
)

from ...core.functor import (
    TransportMechanismFunctor, 
    FunctorMetadata, 
    ProcessCategory, 
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from ...computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from ...computation.operators.iphs import PortSpecIPHS
from ...core.abstractions import (
    StateDict, 
    FluxDict
)
from ...core.state import UpdateItem, UpdateMode
from ...utils.unified_response_functions import (
    validate_positive_values,
    validate_temperature_range,
    validate_unity_range,
)
from ..hydrosphere.utils.psychrometrics import (
    compute_moist_air_density,
    compute_saturation_vapor_pressure,
)

from relic.utils.physical_constants import get_physical_constant


class TurbulenceFunctor(TransportMechanismFunctor):
    """
    Surface layer turbulence functor using Monin-Obukhov Similarity Theory.
    
    Implements comprehensive MOST calculations including friction velocity,
    stability functions, aerodynamic resistance, and heat flux estimations.
    Provides turbulence parameters for ecosystem energy and mass transfer.
    
    Key Features:
        - Full MOST implementation with stability corrections
        - Iterative or analytical solutions for u* and L
        - Roughness length parameterizations for momentum and heat
        - Aerodynamic resistance for scalar transport
        - Optional sensible heat flux calculations
        - Stability parameter diagnostics
    
    Process Category: TRANSPORT_MECHANISM
    Execution Phase: MAIN_PROCESSES
    
    Input Variables:
        - wind_speed [m s-1] at reference height
        - air_temperature [K]
        - surface_temperature [K]
        - reference_height [m] (measurement height)
        - roughness_length_momentum [m] (z0m)
        - displacement_height [m]
        - atmospheric_pressure [Pa]
        - relative_humidity [1]

    Output Variables:
        - friction_velocity [m s-1]
        - aerodynamic_resistance [s m^-1]
        - diagnostics.turbulence.stability_parameter [dimensionless] (z/L)
        - monin_obukhov_length [m]
        - sensible_heat_flux [W m-2]
        - momentum_flux [kg m-1 s-2]
        - diagnostics.turbulence.drag_coefficient [dimensionless]
    
    Coupling Interfaces:
        - Evapotranspiration modules (consume aerodynamic_resistance)
        - Surface energy balance (consume/provide sensible_heat_flux)
        - Scalar transport processes (consume friction_velocity)
    """

    TRANSPORT_TYPE = "turbulent"
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    
    def __init__(self, name: str = "atmospheric_turbulence", config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize turbulence functor with MOST parameters."""
        
        # The simulator replaces this CPU placeholder during registration.
        device = kwargs.get('device', None)
        self.device = device or (torch.device("cpu"))
        
        raw_config = config if config is not None else kwargs.pop('config', None)
        config_dict: Dict[str, Any] = dict(raw_config) if isinstance(raw_config, dict) else {}
        if 'parameters' in config_dict:
            raise ValueError(
                "Turbulence parameters must be supplied through ParameterManager YAML, "
                "not config overrides."
            )
        clean_config: Dict[str, Any] = {}
        dependencies = config_dict.get('dependencies')
        if dependencies is not None:
            clean_config['dependencies'] = dependencies
        
        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Surface layer turbulence using Monin-Obukhov Similarity Theory",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "fm = ln((z-d)/z0m) - psim((z-d)/L) + psim(z0m/L); "
                "fh = ln((z-d)/z0h) - psih((z-d)/L) + psih(z0h/L); "
                "u* = k*U/fm; theta* = k*(Ts-Ta)/fh; "
                "H = rho*cp*u*theta*; "
                "L = -rho*cp*T*u*^3/(k*g*H); rah = fh/(k*u*); tau = rho*u*^2; "
                "rho = (p - (1-eps)*RH*e_s(T))/(R_d*T)"
            ),
            references=[
                "Monin, A.S. & Obukhov, A.M. (1954). Basic laws of turbulent mixing",
                "Businger, J.A. et al. (1971). Flux-profile relationships in the atmospheric surface layer",
                "Garratt, J.R. (1992). The Atmospheric Boundary Layer",
                "Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology",
                "Holton, J.R. & Hakim, G.J. (2012). An Introduction to Dynamic Meteorology, 5th ed., Eqs. 2.63-2.64",
                "Allen, R.G. et al. (1998). FAO Irrigation and Drainage Paper 56, Eqs. 11 and 19",
            ],
            parameters={
                'atmosphere.turbulence.default_reference_height': (
                    'Structural reference height for Hodge audit [m]'
                ),
                'atmosphere.turbulence.roughness_heat_ratio': (
                    'Heat to momentum roughness ratio [dimensionless]'
                ),
                'atmosphere.turbulence.minimum_wind_speed': 'Minimum wind speed threshold [m s-1]',
                'atmosphere.turbulence.minimum_friction_velocity': (
                    'Structural Hodge mobility scale [m s-1]'
                ),
                'atmosphere.turbulence.max_iterations': (
                    'Maximum iteration count for MOST convergence [dimensionless]'
                ),
                'atmosphere.turbulence.convergence_tolerance': (
                    'Convergence tolerance for MOST iteration [dimensionless]'
                ),
                'atmosphere.turbulence.neutral_obukhov_length': (
                    'Canonical solver neutral initialization [m]'
                ),
                'atmosphere.most.beta_stable': 'Stable Businger-Dyer slope [dimensionless]',
                'atmosphere.most.gamma_unstable': (
                    'Unstable Businger-Dyer coefficient [dimensionless]'
                ),
                'atmosphere.most.stable_correction_limit': (
                    'Configured stable Businger-Dyer domain boundary [dimensionless]'
                ),
                'atmosphere.most.unstable_correction_limit': (
                    'Configured unstable Businger-Dyer domain magnitude [dimensionless]'
                ),
                'atmosphere.most.h_min_safeguard': (
                    'Canonical zero-heat diagnostic regularization [W m-2]'
                ),
                'atmosphere.most.monin_obukhov_length_clamp_min': (
                    'Canonical lower diagnostic L bound [m]'
                ),
                'atmosphere.most.monin_obukhov_length_clamp_max': (
                    'Canonical upper diagnostic L bound [m]'
                ),
            },
            input_variables=[
                'wind_speed', 'air_temperature', 'surface_temperature',
                'reference_height', 'roughness_length_momentum', 'displacement_height',
                'atmospheric_pressure', 'relative_humidity',
            ],
            output_variables=[
                'friction_velocity', 'aerodynamic_resistance',
                'diagnostics.turbulence.stability_parameter',
                'monin_obukhov_length', 'sensible_heat_flux', 'momentum_flux',
                'diagnostics.turbulence.drag_coefficient',
            ],
            conservation_laws=['momentum_boundary'],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.MINUTE,
            coupling_interfaces=['evapotranspiration', 'surface_energy_balance', 'scalar_transport'],
            spatial_scale=SpatialScale.GRID_CELL,
            # Process-level classification; per-output diagnostic/flux kinds
            # are owned by the SCM contracts.
            variable_kind=VariableKind.FLUX,
        )
        
        super().__init__(name=name, config=clean_config, metadata=metadata)
        self._load_parameters()

    @staticmethod
    def _scalar_from_value(value: Any) -> float:
        """Convert scalar tensor or numeric value to Python float without explicit host sync calls."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError("Expected scalar tensor value.")
            detached = value.detach().reshape(-1)
            values = detached.tolist()
            if not values:
                raise ValueError("Expected non-empty scalar tensor value.")
            return float(values[0])
        return float(value)

    def _load_parameters(self) -> None:
        """Load turbulence parameters exclusively from the parameter manager."""
        namespace = 'atmosphere.turbulence'

        def to_python(value: Any, caster: Callable[[Any], Any]) -> Any:
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError("Expected scalar parameter tensor.")
                value = self._scalar_from_value(value)
            return caster(value)

        def fetch(key: str, units: str, caster: Callable[[Any], Any]) -> Any:
            value = self.get_parameter_value(
                key,
                namespace=namespace,
                units=units,
                source_order=("manager",),
                required=True,
            )
            return to_python(value, caster)

        self.von_karman_constant = float(get_physical_constant("von_karman_constant"))
        # Boussinesq reference density rho_0 of the structural Hodge mobility
        # only; the runtime MOST fluxes use the moist-air density of the inputs.
        self.reference_air_density = float(
            get_physical_constant("relic.physical.constants.air.density_reference")
        )
        self.tetens_base_pressure = float(
            get_physical_constant("relic.physical.constants.psychrometrics.tetens_base_pressure")
        )
        self.tetens_coefficient_a = float(
            get_physical_constant("relic.physical.constants.psychrometrics.tetens_coefficient_a")
        )
        self.tetens_coefficient_b = float(
            get_physical_constant("relic.physical.constants.psychrometrics.tetens_coefficient_b")
        )
        self.celsius_to_kelvin = float(get_physical_constant("celsius_to_kelvin"))
        self.air_specific_heat_cp = float(get_physical_constant("air_specific_heat_cp"))
        self.gravity_acceleration = float(get_physical_constant("gravity_acceleration"))
        self.default_reference_height = fetch('default_reference_height', 'm', float)
        self.roughness_heat_ratio = fetch('roughness_heat_ratio', 'dimensionless', float)
        self.minimum_wind_speed = fetch('minimum_wind_speed', 'm s-1', float)
        self.minimum_friction_velocity = fetch('minimum_friction_velocity', 'm s-1', float)
        self.max_iterations = fetch('max_iterations', 'dimensionless', int)
        self.convergence_tolerance = fetch('convergence_tolerance', 'dimensionless', float)

    @staticmethod
    def _require_state_tensors(
        state: Mapping[str, Any],
        keys: Tuple[str, ...],
    ) -> Tuple[torch.Tensor, ...]:
        """Return finite floating tensors broadcast to one spatial shape."""
        missing = [key for key in keys if key not in state]
        if missing:
            raise ValueError(
                "Missing required turbulence state inputs: " + ", ".join(missing)
            )

        tensors: List[torch.Tensor] = []
        reference: Optional[torch.Tensor] = None
        for key in keys:
            value = state[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{key} must be a torch.Tensor")
            if not value.is_floating_point():
                raise TypeError(f"{key} must use a floating-point dtype")
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{key} must contain only finite values")
            if reference is None:
                reference = value
            else:
                if value.device != reference.device or value.dtype != reference.dtype:
                    raise ValueError(
                        "Turbulence state inputs must share dtype and device"
                    )
            tensors.append(value)

        try:
            return tuple(torch.broadcast_tensors(*tensors))
        except RuntimeError as exc:
            raise ValueError(
                "Turbulence state inputs must have broadcast-compatible shapes"
            ) from exc

    def _moist_air_density(
        self,
        air_temperature: torch.Tensor,
        atmospheric_pressure: torch.Tensor,
        relative_humidity: torch.Tensor,
    ) -> torch.Tensor:
        """Moist-air density [kg m-3] of the air temperature, pressure and relative humidity inputs."""
        # References: Tetens (1930); Allen et al. (1998), FAO-56 Eqs. 11 and 19;
        #   Holton and Hakim (2012), Eqs. 2.63-2.64 (compute_moist_air_density).
        # Mathematical expression: e = RH * e_s(Ta), e_s = e_0 exp(a Tc / (Tc + b)),
        #   Tc = Ta - K_0; rho = (p - (1 - eps) e) / (R_d Ta).
        # Parameter meanings: RH relative humidity, Ta air temperature, p atmospheric
        #   pressure, e actual and e_s saturation vapour pressure, e_0/a/b/K_0 the
        #   psychrometric physical constants, eps = M_w / M_d, R_d dry-air gas constant.
        # Units: RH [1]; Ta, K_0 [K]; Tc, b [degC]; p, e, e_s, e_0 [Pa]; a [1];
        #   rho [kg m-3].
        # Domain/constraints: 0 <= RH <= 1 raises otherwise; Ta > 0, p > 0 and
        #   0 <= e < p (compute_moist_air_density raises); p and RH have no default.
        # Sign convention: rho is a positive state property, not a flux.
        validate_unity_range(relative_humidity, 'relative_humidity')
        saturation_vapor_pressure = compute_saturation_vapor_pressure(
            air_temperature,
            tetens_base_pressure=self.tetens_base_pressure,
            tetens_coeff_a=self.tetens_coefficient_a,
            tetens_coeff_b=self.tetens_coefficient_b,
            kelvin_offset=self.celsius_to_kelvin,
        )
        return compute_moist_air_density(
            air_temperature,
            atmospheric_pressure,
            relative_humidity * saturation_vapor_pressure,
        )

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare surface-layer turbulence cochain spaces for Phase 8 audit.

        References:
          - Monin and Obukhov (1954), surface-layer similarity.
          - Businger et al. (1971), flux-profile relationships.
        Mathematical expression:
          x_dot = -D_h q_h + G_h u_h, q_h = -star_tau D_h^T e_h.
        Parameter meanings:
          U_h is wind speed, D_h^T U_h is the momentum gradient, and star_tau
          is the non-negative turbulent momentum mobility.
        """

        return (
            CochainSpaceSpec(
                complex_id="atmosphere_surface_layer_turbulence",
                degree=0,
                primality="primal",
                n_dofs=1,
                units="m s-1",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="surface_layer_wind_speed",
                orientation_convention="upward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="atmosphere_surface_layer_turbulence",
                degree=1,
                primality="primal",
                n_dofs=1,
                units="s-1",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="surface_layer_momentum_gradient",
                orientation_convention="upward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="atmosphere_surface_layer_turbulence",
                degree=1,
                primality="dual",
                n_dofs=1,
                units="kg m-1 s-2",
                extensivity="integral",
                physical_domain="atmosphere",
                semantic_name="surface_layer_momentum_flux",
                orientation_convention="upward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders.

        Physical constants are resolved through get_physical_constant() and are
        intentionally excluded from the ParameterManager dependency contract.
        """

        return (
            "atmosphere.turbulence.default_reference_height",
            "atmosphere.turbulence.roughness_heat_ratio",
            "atmosphere.turbulence.minimum_wind_speed",
            "atmosphere.turbulence.minimum_friction_velocity",
            "atmosphere.turbulence.max_iterations",
            "atmosphere.turbulence.convergence_tolerance",
            "atmosphere.turbulence.neutral_obukhov_length",
            "atmosphere.most.beta_stable",
            "atmosphere.most.gamma_unstable",
            "atmosphere.most.stable_correction_limit",
            "atmosphere.most.unstable_correction_limit",
            "atmosphere.most.h_min_safeguard",
            "atmosphere.most.monin_obukhov_length_clamp_min",
            "atmosphere.most.monin_obukhov_length_clamp_max",
        )

    def build_hodge_contract(
        self,
        n_dofs: int,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the positive turbulent momentum mobility Hodge contract.

        References:
          - Stull (1988), first-order turbulent closure with the neutral
            surface-layer eddy viscosity K_m = kappa u* z.
          - Holton and Hakim (2012), Sec. 8.3, Eq. 8.12 (Boussinesq boundary-layer
            momentum equations with the constant reference density rho_0) and
            K theory u'w' = -K_m du/dz; Seinfeld and Pandis (2006), Eq. 2.1.10.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          q_h = -star_tau D_h^T e_h, star_tau = rho_0 kappa u*_min z_ref >= 0.
        Parameter meanings:
          rho_0 is the Boussinesq reference density of dry air
          (relic.physical.constants.air.density_reference, 20 degC and
          101.325 kPa), kappa is the von Karman constant, u*_min is the minimum
          friction velocity, and z_ref is the reference height.
        Units:
          rho_0 [kg m-3]; kappa [1]; u*_min [m s-1]; z_ref [m];
          star_tau [kg m-1 s-1]; D_h^T e_h [s-1]; q_h [kg m-1 s-2].
        Domain/constraints:
          Boussinesq approximation: one state-independent mobility scale for the
          Phase 8 Hodge audit; the runtime stress tau = rho u*^2 uses the
          moist-air density rho of the inputs, never rho_0.
        Sign convention:
          q_h is positive upward (momentum-flux cochain orientation); star_tau
          >= 0 keeps the structural dissipation non-negative.
        """

        n_local = int(n_dofs)
        if n_local < 1:
            raise ValueError("Turbulence Hodge degrees of freedom must be positive.")
        mobility = (
            float(self.reference_air_density)
            * float(self.von_karman_constant)
            * float(self.minimum_friction_velocity)
            * float(self.default_reference_height)
        )
        mobility = max(mobility, 0.0)
        values = torch.full(
            (n_local,),
            mobility,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        gradient_space = CochainSpaceSpec(
            complex_id="atmosphere_surface_layer_turbulence",
            degree=1,
            primality="primal",
            n_dofs=n_local,
            units="s-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="surface_layer_momentum_gradient",
            orientation_convention="upward_positive",
        )
        flux_space = CochainSpaceSpec(
            complex_id="atmosphere_surface_layer_turbulence",
            degree=1,
            primality="dual",
            n_dofs=n_local,
            units="kg m-1 s-2",
            extensivity="integral",
            physical_domain="atmosphere",
            semantic_name="surface_layer_momentum_flux",
            orientation_convention="upward_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_turbulence_momentum_hodge",
            domain=gradient_space,
            codomain=flux_space,
            matrix=torch.diag(values),
            parameter_keys=(
                "atmosphere.turbulence.minimum_friction_velocity",
            ),
            metric_keys=("atmosphere.turbulence.default_reference_height",),
            material_law="surface_layer_turbulent_momentum_mobility",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build turbulence Hodge terms for Phase 8 assembly."""

        # The Hodge acts on the declared state layout unless the complex bundle
        # declares its own degrees of freedom.
        bundle_dofs = getattr(complex_bundle, "n_dofs", None) if complex_bundle is not None else None
        n_dofs = self.declared_forms()[0].n_dofs if bundle_dofs is None else bundle_dofs
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a structural wind-speed audit view."""

        if "wind_speed" not in state:
            return {
                "quadratic_storage_audit": "H_audit = 0.5 * U_h.T U_h",
                "effort_audit": "e_h = U_h",
            }
        wind_speed = state["wind_speed"]
        wind_tensor = wind_speed if isinstance(wind_speed, torch.Tensor) else torch.as_tensor(wind_speed, device=self.device)
        wind_tensor = wind_tensor.to(device=self.device, dtype=torch.float32).reshape(-1)
        # References: Phase 8 structural audit convention.
        # Mathematical expression: H_audit = 0.5 * U_h.T U_h.
        # Parameter meanings: U_h is surface-layer wind speed.
        return {
            "quadratic_storage_audit": 0.5 * torch.sum(wind_tensor * wind_tensor),
            "effort_audit": wind_tensor,
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return non-negative structural turbulent dissipation terms."""

        if "wind_speed" not in state:
            return {
                "dissipation_rate": "sigma_h = grad(U_h).T star_tau grad(U_h)",
                "thermodynamic_force": "grad(U_h)",
            }
        wind_speed = state["wind_speed"]
        wind_tensor = wind_speed if isinstance(wind_speed, torch.Tensor) else torch.as_tensor(wind_speed, device=self.device)
        wind_tensor = wind_tensor.to(device=self.device, dtype=torch.float32).reshape(-1)
        hodge = self.build_hodge_contract(n_dofs=wind_tensor.numel(), strict=True).matrix
        # References: MOST turbulent stress and IPHS dissipative transport form.
        # Mathematical expression: sigma_h = grad(U_h).T star_tau grad(U_h) >= 0.
        # Parameter meanings: star_tau is the non-negative momentum mobility.
        return {
            "dissipation_rate": wind_tensor @ hodge @ wind_tensor,
            "thermodynamic_force": wind_tensor,
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for MOST turbulence closure."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive turbulent momentum mobility."""

        hodge = self.build_hodge_contract(
            n_dofs=self.declared_forms()[0].n_dofs, strict=True
        ).matrix
        if bool(torch.any(hodge < 0.0)):
            raise ValueError("Turbulence mobility must be non-negative in M_h.")
        return {
            "state_vars": ("wind_speed",),
            "J": torch.zeros_like(hodge),
            "M": hodge,
            "ports": self.ports(),
            "open_system_route": "boundary_export",
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the surface-layer momentum exchange port.

        References:
          - Monin and Obukhov (1954), surface-layer stress.
          - Garratt (1992), aerodynamic resistance and momentum exchange.
        Mathematical expression:
          P_port = e_h tau_h, tau_h = rho_air u_*^2.
        Parameter meanings:
          e_h is wind-speed effort and tau_h is momentum flux.
        """

        effort_space = CochainSpaceSpec(
            complex_id="atmosphere_surface_layer_turbulence",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m s-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="surface_layer_wind_speed",
            orientation_convention="upward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="atmosphere_surface_layer_turbulence",
            degree=1,
            primality="dual",
            n_dofs=1,
            units="kg m-1 s-2",
            extensivity="integral",
            physical_domain="atmosphere",
            semantic_name="surface_layer_momentum_flux",
            orientation_convention="upward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="surface_layer_momentum_exchange_port",
                effort_var="wind_speed",
                flow_var="momentum_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="kg s-3",
                positive_power_convention="out_of_system",
                port_kind="momentum",
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

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Audit the open surface momentum boundary relation.

        References:
            - Monin and Obukhov (1954), surface-layer similarity theory.
            - Garratt (1992), surface stress and friction velocity.
        Mathematical expression:
            R_tau = max(|tau - rho_air u_star^2|
                        / max(|tau|, |rho_air u_star^2|)).
        Parameter meanings:
            tau is surface stress [kg m-1 s-2], rho_air is the moist-air
            density [kg m-3] of the input air temperature, pressure and
            relative humidity, and u_star is friction velocity [m s-1]. A unit
            scale is used only for a double-zero denominator.
        Open-system route:
            Momentum leaves through the declared surface boundary port. This
            constitutive audit does not claim a closed atmospheric momentum
            or coupled surface-energy balance.
        """

        def require_tensor(key: str) -> torch.Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing conservation output '{key}'")
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Turbulence conservation output '{key}' "
                    "must be a torch.Tensor."
                )
            if value.numel() == 0:
                raise ValueError(
                    f"Turbulence conservation output '{key}' "
                    "must be non-empty."
                )
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"Turbulence conservation output '{key}' "
                    "must be floating."
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    f"Turbulence conservation output '{key}' "
                    "must be finite."
                )
            return value

        friction_velocity = require_tensor('friction_velocity')
        momentum_flux = require_tensor('momentum_flux')
        if friction_velocity.shape != momentum_flux.shape:
            raise ValueError(
                "Turbulence conservation outputs must have identical shapes."
            )
        if (
            friction_velocity.device != momentum_flux.device
            or friction_velocity.dtype != momentum_flux.dtype
        ):
            raise ValueError(
                "Turbulence conservation outputs must share dtype and device."
            )

        air_temperature, atmospheric_pressure, relative_humidity = (
            self._require_state_tensors(
                input_state,
                ('air_temperature', 'atmospheric_pressure', 'relative_humidity'),
            )
        )
        air_density = self._moist_air_density(
            air_temperature, atmospheric_pressure, relative_humidity
        ).to(dtype=friction_velocity.dtype, device=friction_velocity.device)
        expected_momentum_flux = air_density * friction_velocity.square()
        scale = torch.maximum(
            torch.abs(momentum_flux),
            torch.abs(expected_momentum_flux),
        )
        scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
        residual = torch.abs(momentum_flux - expected_momentum_flux) / scale
        return {
            'momentum_boundary': float(torch.max(residual).detach().item())
        }

    def forward(self,
        # References: Monin and Obukhov (1954); Businger et al. (1971); Garratt (1992); Stull (1988).
        # Mathematical expression: fm = ln((z-d)/z0m) - psim((z-d)/L) + psim(z0m/L);
        #   fh = ln((z-d)/z0h) - psih((z-d)/L) + psih(z0h/L); u* = k U / fm;
        #   theta* = k (Ts - Ta) / fh; H = rho cp u* theta*; L = -rho cp T u*^3 / (k g H);
        #   rah = fh / (k u*).
        # Parameter meanings: z reference height, d displacement height, z0m/z0h roughness
        #   lengths, L Obukhov length, k von Karman constant, U wind speed, Ts and Ta surface
        #   and air temperature, rho the moist-air density of Ta, p and RH, cp heat capacity.
        # Units: u*, U [m s-1]; z, d, z0m, z0h, L [m]; rah [s m-1]; H [W m-2];
        #   momentum flux [kg m-1 s-2]; stability terms [1]; T [K]; p [Pa]; RH [1];
        #   rho [kg m-3].
        # Domain/constraints: iterative MOST closure on finite required inputs with positive
        #   heights and roughness lengths; stability functions follow the configured forms;
        #   0 <= RH <= 1 and 0 <= RH e_s(Ta) < p; outputs are REPLACE diagnostics.
        # Sign convention: H positive upward from the surface; L < 0 unstable, L > 0 stable.
                state: StateDict,
                time: float,
                dt: float,
                **kwargs) -> FluxDict:
        """
        Compute atmospheric turbulence parameters using MOST.
        
        This method calculates friction velocity, aerodynamic resistance,
        and stability parameters based on wind speed, temperature gradients,
        and surface roughness using Monin-Obukhov Similarity Theory.
        
        Parameters:
            state : StateDict
                Current system state containing meteorological variables
                Required keys: wind_speed, air_temperature, surface_temperature,
                               reference_height, roughness_length_momentum,
                               displacement_height, atmospheric_pressure [Pa],
                               relative_humidity [1]
            time : float
                Current simulation time [s]
            dt : float
                Time step [s]
                
        Returns:
            FluxDict: Turbulence parameters and fluxes with the following keys and units:
                - friction_velocity: Friction velocity (u*) [m s-1]
                - aerodynamic_resistance: Scalar heat and mass resistance (rah) [s m^-1]
                - diagnostics.turbulence.stability_parameter: z/L [dimensionless]
                - monin_obukhov_length: Monin-Obukhov length scale (L) [m]
                - sensible_heat_flux: Sensible heat flux (H) [W m-2]
                - momentum_flux: Surface stress [kg m-1 s-2]
                - diagnostics.turbulence.drag_coefficient: Neutral CD [dimensionless]
        
        Raises:
            ValueError: If required inputs are missing, non-finite, or invalid
        """
        input_keys = (
            'wind_speed',
            'air_temperature',
            'surface_temperature',
            'reference_height',
            'roughness_length_momentum',
            'displacement_height',
            'atmospheric_pressure',
            'relative_humidity',
        )
        (
            wind_speed,
            air_temp,
            surface_temp,
            reference_height,
            roughness_momentum,
            displacement_height,
            atmospheric_pressure,
            relative_humidity,
        ) = self._require_state_tensors(state, input_keys)

        # References: Monin and Obukhov (1954); Businger et al. (1971).
        # Mathematical expression: u*=kU/fm, H=rho cp u* theta*,
        # L=-rho cp T u*^3/(k g H), and rah=fh/(k u*).
        # Parameter meanings: z is reference height, d is displacement height,
        # z0m/z0h are roughness lengths, rho is the moist-air density of the
        # inputs (_moist_air_density), and k/cp/g use the physical SSOT.
        # Units: U, u* [m s-1]; z, d, z0m, z0h, L [m]; H [W m-2]; rah [s m-1];
        #   rho [kg m-3]; cp [J kg-1 K-1]; T [K]; p [Pa]; RH [1].
        # Domain/constraints: U >= minimum_wind_speed; z - d > z0m and z0h;
        #   positive roughness lengths; 0 <= RH <= 1 and 0 <= RH e_s(T) < p.
        # Sign convention: H > 0 upward from the surface; L < 0 unstable.
        # Open-system route: H and momentum flux are explicit surface boundary ports.
        validate_positive_values(wind_speed, 'wind_speed')
        validate_temperature_range(air_temp)
        validate_temperature_range(surface_temp)
        minimum_wind = wind_speed.new_tensor(float(self.minimum_wind_speed))
        comparison_tolerance = torch.finfo(wind_speed.dtype).eps * torch.maximum(
            torch.ones_like(minimum_wind),
            torch.abs(minimum_wind),
        )
        if bool(torch.any(wind_speed < minimum_wind - comparison_tolerance)):
            raise ValueError(
                "wind_speed is below the ParameterManager minimum_wind_speed"
            )
        if bool(torch.any(reference_height <= 0.0)):
            raise ValueError("reference_height must be positive")
        if bool(torch.any(roughness_momentum <= 0.0)):
            raise ValueError("roughness_length_momentum must be positive")
        if bool(torch.any(displacement_height < 0.0)):
            raise ValueError("displacement_height must be non-negative")
        air_density = self._moist_air_density(
            air_temp, atmospheric_pressure, relative_humidity
        )

        # z0h follows one declared closure path; no state alias or fallback is used.
        roughness_heat = roughness_momentum * float(self.roughness_heat_ratio)
        effective_height = reference_height - displacement_height
        if bool(torch.any(effective_height <= roughness_momentum)):
            raise ValueError(
                "reference_height - displacement_height must exceed "
                "roughness_length_momentum"
            )
        if bool(torch.any(effective_height <= roughness_heat)):
            raise ValueError(
                "reference_height - displacement_height must exceed derived "
                "roughness_length_heat"
            )
        
        u_star, L, sensible_heat_flux = iterative_most_solution(
            wind_speed,
            air_temp,
            surface_temp,
            reference_height,
            roughness_momentum,
            roughness_heat,
            displacement_height,
            max_iterations=int(self.max_iterations),
            tolerance=float(self.convergence_tolerance),
            k=float(self.von_karman_constant),
            rho=air_density,
            cp=float(self.air_specific_heat_cp),
            g=float(self.gravity_acceleration),
        )
        stability_parameter = most_zeta(
            reference_height,
            L,
            displacement_height,
        )
        aerodynamic_resistance = most_aerodynamic_resistance(
            u_star,
            reference_height,
            roughness_heat,
            stability_parameter,
            displacement_height,
            k=float(self.von_karman_constant),
        )
        
        # Calculate drag coefficient (neutral conditions)
        # Note: This is the neutral drag coefficient CD_n = (k/ln(z/z0))^2
        # For stability-corrected drag coefficient, use:
        # CD_corrected = (k/(ln(z/z0) - psim))^2
        # However, we provide the neutral value for diagnostic purposes
        # Drag coefficient neutral: CD = (k/ln(z/z0))^2 , with k from parameter system
        k = torch.as_tensor(
            self.von_karman_constant,
            dtype=wind_speed.dtype,
            device=wind_speed.device,
        )
        ln_neutral = torch.log(effective_height / roughness_momentum)
        drag_coefficient = (k / ln_neutral)**2
        
        # References: Stull (1988), Sec. 2.10; Garratt (1992), Ch. 3.
        # Mathematical expression: tau = rho_air * u_star^2.
        # Parameter meanings: rho_air is the moist-air density of the inputs,
        #   u_star is friction velocity.
        # Units: rho_air [kg m-3]; u_star [m s-1]; tau [kg m-1 s-2].
        # Domain/constraints: rho_air > 0 from Ta > 0, p > 0, 0 <= e < p.
        # Sign convention: tau is the magnitude of the downward momentum flux.
        momentum_flux = most_momentum_flux(u_star, air_density)

        # References: Monin and Obukhov (1954); Businger et al. (1971); Stull (1988).
        # Mathematical expression: scalar port flow coefficient g_a = 1 / r_a,
        # with r_a from MOST and momentum flux tau = rho_air u_star^2.
        # Parameter meanings: r_a is aerodynamic resistance, u_star is friction
        # velocity, rho_air is the moist-air density of the inputs, and
        # stability_parameter is z/L.
        # Units: r_a [s m-1]; g_a [m s-1]; u_star [m s-1]; rho_air [kg m-3];
        #   tau [kg m-1 s-2]; z/L [1].
        # Domain/constraints: every output REPLACEs a per-step diagnostic of the
        #   validated finite inputs.
        # Sign convention: H > 0 upward; tau is a positive stress magnitude.
        return {
            'friction_velocity': UpdateItem(
                value=u_star,
                mode=UpdateMode.REPLACE,
                unit='m s-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'aerodynamic_resistance': UpdateItem(
                value=aerodynamic_resistance,
                mode=UpdateMode.REPLACE,
                unit='s m-1',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.turbulence.stability_parameter': UpdateItem(
                value=stability_parameter,
                mode=UpdateMode.REPLACE,
                unit='dimensionless',
                per_timestep=True,
                source_functor=self.name,
            ),
            'monin_obukhov_length': UpdateItem(
                value=L,
                mode=UpdateMode.REPLACE,
                unit='m',
                per_timestep=True,
                source_functor=self.name,
            ),
            'sensible_heat_flux': UpdateItem(
                value=sensible_heat_flux,
                mode=UpdateMode.REPLACE,
                unit='W m-2',
                per_timestep=True,
                source_functor=self.name,
            ),
            'momentum_flux': UpdateItem(
                value=momentum_flux,
                mode=UpdateMode.REPLACE,
                unit='kg m-1 s-2',
                per_timestep=True,
                source_functor=self.name,
            ),
            'diagnostics.turbulence.drag_coefficient': UpdateItem(
                value=drag_coefficient,
                mode=UpdateMode.REPLACE,
                unit='dimensionless',
                per_timestep=True,
                source_functor=self.name,
            ),
        }
__all__ = ['TurbulenceFunctor']


