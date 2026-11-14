"""
Atmospheric Boundary Layer Eddy Diffusion Transport Functor

This module implements atmospheric boundary-layer (ABL) eddy diffusion for gaseous
species. The production calculation is a single K-theory relaxation closure whose
eddy diffusivity is constrained by Monin-Obukhov Similarity Theory (MOST).

Mathematical Expression:
    Species transport equation: partialC/partialt = partial/partialz[K(z) partialC/partialz]
    where C is species mixing ratio, K is eddy diffusivity, h is boundary layer height
    
    Prandtl mixing length: K = l^2|partialV/partialz|, where l = kappaz (surface layer)
    MOST formulation: K = kappau*z/phi_h(z/L)
    
    Reduced box closure: dC/dt = K/h^2 * (C_background - C)

Scientific References:
    - Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology
    - Garratt, J.R. (1992). The Atmospheric Boundary Layer
    - Businger, J.A. et al. (1971). Flux-profile relationships in the atmospheric surface layer
    - Prandtl, L. (1925). Ueber die ausgebildete Turbulenz. ZAMM
    - Monin, A.S. & Obukhov, A.M. (1954). Basic laws of turbulent mixing

Implementation Notes:
    - Default disabled (opt-in activation)
    - CFL stability validation with fail-fast rejection
    - Numerical stability clamps for extreme conditions
    - Integration with existing TurbulenceFunctor diagnostics
    - Conservative transport formulation
    - Multi-species support with configurable selection
"""

import logging
import math
from typing import Dict, List, Any, Literal, Optional, Union, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor

# Import unified transport operators
# Import unified transport operators for spatial calculations
from ....computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from ....computation.operators.iphs import PortSpecIPHS
from ....computation.operators.functional import compute_diffusion_stability

# Import functions that are not yet available in computation layer
# Note: compute_u_star and compute_L are Monin-Obukhov specific functions, may need to be implemented in unified operators

# Import bridge class and types
from ....core.functor import (
    TransportMechanismFunctor,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    ConfigDict,
    DependencyList,
    TemporalScale,
)
from ....core.abstractions import (
    StateDict, 
    FluxDict
)
from ....core.state import UpdateItem, UpdateMode
from ....parameters.namespaces import ParameterNamespace, resolve_namespace
from relic.utils.physical_constants import get_von_karman
from ..most import phi_h as most_phi_h

TRANSPORT_NUM_NS = resolve_namespace(ParameterNamespace.TRANSPORT_NUMERICAL)
ATMOS_TRANSPORT_NS = resolve_namespace(ParameterNamespace.ATMOSPHERE_TRANSPORT)
ATMOS_MOST_NS = resolve_namespace(ParameterNamespace.ATMOSPHERE_MOST)

logger = logging.getLogger(__name__)


def _to_float(value: Any) -> float:
    """Convert parameter manager outputs to Python float."""
    if isinstance(value, Tensor):
        detached = value.detach()
        if detached.numel() != 1:
            raise ValueError("Expected one scalar floating-point parameter value.")
        return float(detached.item())
    return float(value)


def _to_bool(value: Any) -> bool:
    """Convert parameter manager outputs to Python bool."""
    if isinstance(value, Tensor):
        detached = value.detach()
        if detached.numel() != 1:
            raise ValueError("Expected one scalar Boolean parameter value.")
        return bool(detached.item())
    return bool(value)


def _normalize_species(value: Optional[Any]) -> Optional[List[str]]:
    """Validate the ordered target-species schema from ParameterManager."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError("target_species must be an ordered list or tuple.")


@dataclass
class EddyDiffusionConfig:
    """
    Configuration container for eddy diffusion activation.
    
    Scientific and numerical values are owned by the ParameterManager and
    physical-constants SSOT, not by this activation container.
    
    References:
        - Stull, R.B. (1988). Boundary layer parameterizations
        - Garratt, J.R. (1992). Turbulent transport parameters
    """
    # Transport activation control
    enabled: Optional[bool] = None
    
class EddyDiffusionBLMFunctor(TransportMechanismFunctor):
    """
    Boundary Layer Eddy Diffusion Transport Mechanism Functor
    
    Implements atmospheric boundary layer eddy diffusion transport for gaseous species
    using one K-theory closure with Monin-Obukhov similarity corrections.
    
    Transport Formulation:
        Full transport: dC/dt = partial/partialz[K partialC/partialz]
        K-theory diffusion: K = kappau*z/phi_h(z/L) with stability corrections
        Reduced closure: dC/dt = K/h^2 * (C_background - C)
        
        Where:
        - C: species mixing ratio [mol mol-1]
        - K: eddy diffusivity [m^2 s-1]
        - h: boundary layer height [m]
        - u*: friction velocity [m s-1]
        - L: Monin-Obukhov length [m]
        - phi_h: MOST stability function [dimensionless]
    
    Stability Functions (MOST):
        Stable (z/L >= 0): phi_h = 1 + beta_h(z/L) where beta_h ~= 5.0
        Unstable (z/L < 0): phi_h = (1 - gamma_h(z/L))^(-1/2) where gamma_h ~= 16.0
        
    CFL Stability:
        sigma = K*dt/h_min^2 <= sigma_max for explicit integration
        
    Conservation:
        Open top-boundary ledger for each species:
        DeltaC = dt*K/h^2*(C_background - C)
        
    Attributes:
        config : EddyDiffusionConfig
            Transport configuration parameters
        device : torch.device
            Computation device (CPU/GPU)
        
    References:
        - Businger, J.A., Wyngaard, J.C., Izumi, Y., & Bradley, E.F. (1971). 
          Flux-profile relationships in the atmospheric surface layer. J. Atmos. Sci., 28, 181-189
        - Holtslag, A.A.M. & De Bruin, H.A.R. (1988). Applied modeling of the nighttime 
          surface energy balance over land. J. Appl. Meteor., 27, 689-704
        - Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology. Kluwer Academic
    """

    TRANSPORT_TYPE = "boundary_layer_diffusion"
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    PHYSICAL_CONSTANT_KEYS = (
        "relic.physical.constants.turbulence.von_karman_constant",
    )
    SPECIES_MAP = {
        'no': 'no_mixing_ratio',
        'no2': 'no2_mixing_ratio',
        'o3': 'o3_mixing_ratio',
        'isoprene': 'isoprene_mixing_ratio',
    }
    SURFACE_LAYER_SOURCES = ('atmospheric_turbulence', 'external_forcing')
    
    def __init__(self,
                 name: str = "eddy_diffusion_blm",
                 config: Optional[ConfigDict] = None,
                 transport_config: Optional[EddyDiffusionConfig] = None,
                 device: Optional[torch.device] = None,
                 *,
                 surface_layer_source: Literal[
                     'atmospheric_turbulence', 'external_forcing'
                 ] = 'atmospheric_turbulence') -> None:
        """
        Initialize boundary layer eddy diffusion transport functor.
        
        Parameters:
            name : str
                Functor instance name
            config : Optional[ConfigDict]
                General functor configuration
            transport_config : Optional[EddyDiffusionConfig]
                Transport-specific configuration
            surface_layer_source : str
                GraphIR selection field, not a parameter: the provider of
                friction_velocity and monin_obukhov_length, either the
                atmospheric_turbulence Functor (a required dependency) or
                external forcing such as an observed tower series. The law
                evaluated in forward is the same for both.
                
        Mathematical Expression:
            Initialization: F_init: Config -> TransportFunctor
            Parameter space: Theta = {kappa, h_pbl, K_min, K_max, z/L_lim, ...}
        """
        if surface_layer_source not in self.SURFACE_LAYER_SOURCES:
            raise ValueError(
                f"surface_layer_source must be one of {self.SURFACE_LAYER_SOURCES}, "
                f"got {surface_layer_source!r}."
            )
        # Default configuration
        if config is None:
            config = {
                'parameters': {},
                'dependencies': (
                    ['atmospheric_turbulence']
                    if surface_layer_source == 'atmospheric_turbulence'
                    else []
                ),
            }
        
        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Atmospheric boundary layer eddy diffusion transport with MOST corrections",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                r"$\frac{\partial C}{\partial t} = \frac{\partial}{\partial z}\left[K(z,L) "
                r"\frac{\partial C}{\partial z}\right]$; reduced closure "
                r"$\dot C = K h^{-2}(C_{bg}-C)$, where "
                r"$K = \frac{\kappa u_* z}{\phi_h(z/L)}$"
            ),
            references=[
                "Businger, J.A. et al. (1971). Flux-profile relationships in surface layer. J. Atmos. Sci., 28, 181-189",
                "Stull, R.B. (1988). An Introduction to Boundary Layer Meteorology. Kluwer Academic",
                "Garratt, J.R. (1992). The Atmospheric Boundary Layer. Cambridge University Press",
                "Monin, A.S. & Obukhov, A.M. (1954). Basic laws of turbulent mixing. Trudy Geofiz. Inst. AN SSSR"
            ],
            parameters={
                'atmosphere.transport.enabled': 'Opt-in process activation flag [dimensionless]',
                'atmosphere.transport.target_species': (
                    'Canonical transported species list [dimensionless]'
                ),
                'atmosphere.transport.k_min': 'Minimum eddy diffusivity for stability [m2 s-1]',
                'atmosphere.transport.k_max': 'Maximum eddy diffusivity limit [m2 s-1]',
                'atmosphere.transport.z_over_L_limit': (
                    'Absolute MOST stability parameter bound [dimensionless]'
                ),
                'atmosphere.transport.ustar_min': 'Minimum friction velocity [m s-1]',
                'atmosphere.transport.ustar_max': 'Maximum friction velocity [m s-1]',
                'atmosphere.transport.pbl_height_min': 'Minimum accepted PBL height [m]',
                'atmosphere.transport.pbl_height_max': 'Maximum accepted PBL height [m]',
                'atmosphere.transport.background_no_mixing_ratio': (
                    'Top-boundary NO mixing ratio [mol mol-1]'
                ),
                'atmosphere.transport.background_no2_mixing_ratio': (
                    'Top-boundary NO2 mixing ratio [mol mol-1]'
                ),
                'atmosphere.transport.background_o3_mixing_ratio': (
                    'Top-boundary O3 mixing ratio [mol mol-1]'
                ),
                'atmosphere.transport.background_isoprene_mixing_ratio': (
                    'Top-boundary isoprene mixing ratio [mol mol-1]'
                ),
                'transport.numerical.cfl_safety_factor': (
                    'Safety factor on the CFL stability limits [dimensionless]'
                ),
                'transport.numerical.max_diffusion_stability': (
                    'Maximum explicit diffusion stability number [dimensionless]'
                ),
                'atmosphere.most.beta_stable': (
                    'Stable MOST scalar-gradient coefficient [dimensionless]'
                ),
                'atmosphere.most.gamma_unstable': (
                    'Unstable MOST scalar-gradient coefficient [dimensionless]'
                ),
                'atmosphere.most.stable_correction_limit': (
                    'Shared positive MOST operational limit [dimensionless]'
                ),
                'atmosphere.most.unstable_correction_limit': (
                    'Shared absolute negative MOST operational limit [dimensionless]'
                ),
            },
            input_variables=[
                'no_mixing_ratio', 'no2_mixing_ratio', 'o3_mixing_ratio', 
                'isoprene_mixing_ratio', 'pbl_height', 'friction_velocity',
                'monin_obukhov_length'
            ],
            output_variables=[
                'no_mixing_ratio', 'no2_mixing_ratio', 'o3_mixing_ratio',
                'isoprene_mixing_ratio',
                'diagnostics.eddy_diffusion.enabled',
                'diagnostics.eddy_diffusion.eddy_diffusivity',
                'diagnostics.eddy_diffusion.pbl_height',
                'diagnostics.eddy_diffusion.friction_velocity',
                'diagnostics.eddy_diffusion.monin_obukhov_length',
                'diagnostics.eddy_diffusion.stability_number',
                'diagnostics.eddy_diffusion.stability_parameter_z_over_L',
                'diagnostics.eddy_diffusion.eddy_diffusivity_clamped',
                'diagnostics.eddy_diffusion.stability_parameter_clipping_residual'
            ],
            conservation_laws=['mass'],  # Mass conservation for each species
            spatial_dimensions=1,  # 1D vertical transport
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=['turbulence', 'chemistry', 'dry_deposition'], spatial_scale=SpatialScale.GRID_CELL, variable_kind=VariableKind.STATE
        )
        
        # Initialize via cooperative inheritance with proper parameter passing
        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
            device=device,
        )
        self.surface_layer_source = surface_layer_source
        
        # Transport-specific configuration
        if transport_config is None:
            transport_config = EddyDiffusionConfig()

        self.config_transport = self._hydrate_transport_config(transport_config)
        self.target_species = self._load_target_species()
        self._runtime_parameters = MappingProxyType(
            self._load_runtime_parameters()
        )
        # The inherited scalar diffusivity stays unset (None): canonical K is
        # state-dependent and is evaluated only in forward().

        # Cache ParameterManager-backed bounds and boundary forcings.
        self._transport_parameters = MappingProxyType(
            self._load_transport_parameters()
        )

        # Numerical diagnostics storage
        self._diagnostics: Dict[str, Union[float, Tensor]] = {}

        # Species mapping for flexibility
        self.species_map = dict(self.SPECIES_MAP)

        logger.info(f"Initialized {self.__class__.__name__} '{name}' "
                   f"(enabled={self.config_transport.enabled})")

    def _hydrate_transport_config(self, config: EddyDiffusionConfig) -> EddyDiffusionConfig:
        """Populate the activation flag from the canonical parameter path."""
        namespace = ATMOS_TRANSPORT_NS

        if config.enabled is None:
            config.enabled = _to_bool(
                self.get_parameter_value("enabled", namespace=namespace, required=True)
            )
        return config

    def _load_target_species(self) -> tuple[str, ...]:
        """Load and validate the immutable canonical species schema."""
        namespace = ATMOS_TRANSPORT_NS
        species_raw = self.get_parameter_value(
            "target_species", namespace=namespace, required=True
        )
        target_species = _normalize_species(species_raw)
        if target_species is None:
            raise ValueError("target_species is required for EddyDiffusionConfig.")
        if target_species != list(self.SPECIES_MAP):
            raise ValueError(
                "Eddy diffusion target_species must be the canonical ordered set: "
                + ", ".join(self.SPECIES_MAP)
            )
        return tuple(target_species)

    def _load_runtime_parameters(self) -> Dict[str, float]:
        """Load the read-only numerical parameter snapshot from SSOT owners."""
        namespace = ATMOS_TRANSPORT_NS
        values = {
            "von_karman_constant": _to_float(get_von_karman()),
            "k_min": _to_float(
            self.get_parameter_value("k_min", namespace=namespace, required=True)
            ),
            "k_max": _to_float(
            self.get_parameter_value("k_max", namespace=namespace, required=True)
            ),
            "z_over_L_limit": _to_float(
                self.get_parameter_value(
                    "z_over_L_limit", namespace=namespace, required=True
                )
            ),
            "ustar_min": _to_float(
            self.get_parameter_value("ustar_min", namespace=namespace, required=True)
            ),
            "ustar_max": _to_float(
            self.get_parameter_value("ustar_max", namespace=namespace, required=True)
            ),
            "cfl_factor": _to_float(
                self.get_parameter_value(
                    "cfl_safety_factor",
                    namespace=TRANSPORT_NUM_NS,
                    required=True,
                )
            ),
        }
        if not 0.3 <= values["von_karman_constant"] <= 0.5:
            raise ValueError(
                "Physical constant von_karman_constant is outside [0.3, 0.5]."
            )
        if values["k_min"] <= 0.0 or values["k_max"] <= values["k_min"]:
            raise ValueError(
                f"Invalid ParameterManager K bounds: "
                f"k_min={values['k_min']}, k_max={values['k_max']}"
            )
        if not 0.0 < values["cfl_factor"] <= 1.0:
            raise ValueError(
                "ParameterManager cfl_safety_factor must be in (0, 1]."
            )
        return values

    def _load_transport_parameters(self) -> Dict[str, float]:
        """
        Load the Eddy diffusion process parameters: the PBL validity bounds and the
        background mixing ratios C_bg of the relaxation law r = K / h^2 (C_bg - C).

        Returns
        -------
        Dict[str, float]
            Dictionary containing PBL bounds and species background mixing ratios.
        """
        pbl_clamp_min = self.get_parameter_value(
            "pbl_height_min",
            namespace=ATMOS_TRANSPORT_NS,
            source_order=("manager",), required=True)
        pbl_clamp_max = self.get_parameter_value(
            "pbl_height_max",
            namespace=ATMOS_TRANSPORT_NS,
            source_order=("manager",), required=True)
        bg_no = self.get_parameter_value(
            "background_no_mixing_ratio",
            namespace=ATMOS_TRANSPORT_NS,
            source_order=("manager",), required=True)
        bg_no2 = self.get_parameter_value(
            "background_no2_mixing_ratio",
            namespace=ATMOS_TRANSPORT_NS,
            source_order=("manager",), required=True)
        bg_o3 = self.get_parameter_value(
            "background_o3_mixing_ratio",
            namespace=ATMOS_TRANSPORT_NS,
            source_order=("manager",), required=True)
        bg_isoprene = self.get_parameter_value(
            "background_isoprene_mixing_ratio",
            namespace=ATMOS_TRANSPORT_NS,
            source_order=("manager",), required=True)

        return {
            "pbl_height_min": _to_float(pbl_clamp_min),
            "pbl_height_max": _to_float(pbl_clamp_max),
            "background_no_mixing_ratio": _to_float(bg_no),
            "background_no2_mixing_ratio": _to_float(bg_no2),
            "background_o3_mixing_ratio": _to_float(bg_o3),
            "background_isoprene_mixing_ratio": _to_float(bg_isoprene),
        }
    
    def get_dependencies(self) -> DependencyList:
        """
        Declare the canonical turbulence producer dependency.

        Returns:
            DependencyList: Required functors that must execute first

        Dependencies:
            - atmospheric_turbulence: For u* and L diagnostics via MOST

        Notes:
            Dependencies are architectural decisions, not configurable parameters.
            With surface_layer_source = external_forcing u* and L are forcing and
            no producer exists; a missing value raises in forward.
        """
        if self.surface_layer_source == 'external_forcing':
            return self._merge_declared_dependencies([])
        return self._merge_declared_dependencies(['atmospheric_turbulence'])

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare boundary-layer eddy diffusion cochain spaces for Phase 8 audit.

        References:
          - Stull (1988), K-theory turbulent diffusion.
          - Businger et al. (1971), MOST surface-layer transport.
        Mathematical expression:
          x_dot = -D_h q_h + G_h u_h, q_h = -star_K D_h^T e_h.
        Parameter meanings:
          C_h is the species mixing-ratio cochain, D_h^T C_h is the vertical
          gradient, and star_K is the non-negative eddy diffusivity Hodge term.
        """

        n_species = len(self.target_species)
        return (
            CochainSpaceSpec(
                complex_id="atmosphere_boundary_layer_eddy_diffusion",
                degree=0,
                primality="primal",
                n_dofs=n_species,
                units="mol mol-1",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="boundary_layer_species_mixing_ratio",
                orientation_convention="upward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="atmosphere_boundary_layer_eddy_diffusion",
                degree=1,
                primality="primal",
                n_dofs=n_species,
                units="mol mol-1 m-1",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="boundary_layer_species_gradient",
                orientation_convention="upward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="atmosphere_boundary_layer_eddy_diffusion",
                degree=1,
                primality="dual",
                n_dofs=n_species,
                units="mol mol-1 m s-1",
                extensivity="integral",
                physical_domain="atmosphere",
                semantic_name="boundary_layer_turbulent_flux",
                orientation_convention="upward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "atmosphere.transport.enabled",
            "atmosphere.transport.target_species",
            "atmosphere.transport.k_min",
            "atmosphere.transport.k_max",
            "atmosphere.transport.z_over_L_limit",
            "atmosphere.transport.ustar_min",
            "atmosphere.transport.ustar_max",
            "atmosphere.transport.pbl_height_min",
            "atmosphere.transport.pbl_height_max",
            "atmosphere.transport.background_no_mixing_ratio",
            "atmosphere.transport.background_no2_mixing_ratio",
            "atmosphere.transport.background_o3_mixing_ratio",
            "atmosphere.transport.background_isoprene_mixing_ratio",
            "transport.numerical.cfl_safety_factor",
            "transport.numerical.max_diffusion_stability",
            "atmosphere.most.beta_stable",
            "atmosphere.most.gamma_unstable",
            "atmosphere.most.stable_correction_limit",
            "atmosphere.most.unstable_correction_limit",
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
        """Build the positive eddy diffusivity Hodge contract.

        References:
          - Stull (1988), gradient transport closure.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          q_h = -star_K D_h^T e_h, star_K = K_h >= 0.
        Parameter meanings:
          K_h is the ParameterManager-backed eddy diffusivity bounded by k_min and k_max.
        """

        n_local = int(n_dofs) if n_dofs is not None else len(self.target_species)
        if n_local < 1:
            raise ValueError("Eddy diffusion Hodge degrees of freedom must be positive.")
        mobility = self._runtime_parameters["k_min"]
        if mobility <= 0.0:
            raise ValueError("Eddy diffusion Hodge mobility must be positive.")
        values = torch.full(
            (n_local,),
            mobility,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        gradient_space = CochainSpaceSpec(
            complex_id="atmosphere_boundary_layer_eddy_diffusion",
            degree=1,
            primality="primal",
            n_dofs=n_local,
            units="mol mol-1 m-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="boundary_layer_species_gradient",
            orientation_convention="upward_positive",
        )
        flux_space = CochainSpaceSpec(
            complex_id="atmosphere_boundary_layer_eddy_diffusion",
            degree=1,
            primality="dual",
            n_dofs=n_local,
            units="mol mol-1 m s-1",
            extensivity="integral",
            physical_domain="atmosphere",
            semantic_name="boundary_layer_turbulent_flux",
            orientation_convention="upward_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_eddy_diffusion_hodge",
            domain=gradient_space,
            codomain=flux_space,
            matrix=torch.diag(values),
            parameter_keys=(
                "atmosphere.transport.k_min",
                "atmosphere.transport.k_max",
            ),
            metric_keys=(
                "atmosphere.transport.pbl_height_min",
                "atmosphere.transport.pbl_height_max",
            ),
            material_law="boundary_layer_eddy_diffusivity",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build eddy diffusion Hodge terms for Phase 8 assembly."""

        n_dofs = getattr(complex_bundle, "n_dofs", None) if complex_bundle is not None else None
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return a structural boundary-layer concentration audit view."""

        vectors: List[Tensor] = []
        for species in self.target_species:
            state_var = self.species_map.get(species, f"{species}_mixing_ratio")
            if state_var in state:
                value = state[state_var]
                tensor = value if isinstance(value, Tensor) else torch.as_tensor(value, device=self.device)
                vectors.append(tensor.to(device=self.device, dtype=torch.float32).reshape(-1))
        if not vectors:
            return {
                "quadratic_storage_audit": "H_audit = 0.5 * C_h.T C_h",
                "effort_audit": "e_h = C_h",
            }
        # References: Phase 8 structural audit convention.
        # Mathematical expression: H_audit = 0.5 * C_h.T C_h.
        # Parameter meanings: C_h is the boundary-layer species mixing-ratio cochain.
        concentration = torch.cat(vectors)
        return {
            "quadratic_storage_audit": 0.5 * torch.sum(concentration * concentration),
            "effort_audit": concentration,
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return non-negative structural eddy diffusion dissipation terms."""

        vectors: List[Tensor] = []
        for species in self.target_species:
            state_var = self.species_map.get(species, f"{species}_mixing_ratio")
            if state_var in state:
                value = state[state_var]
                tensor = value if isinstance(value, Tensor) else torch.as_tensor(value, device=self.device)
                vectors.append(tensor.to(device=self.device, dtype=torch.float32).reshape(-1))
        if not vectors:
            return {
                "dissipation_rate": "sigma_h = grad(C_h).T star_K grad(C_h)",
                "thermodynamic_force": "grad(C_h)",
            }
        concentration = torch.cat(vectors)
        hodge = self.build_hodge_contract(n_dofs=concentration.numel(), strict=True).matrix
        # References: K-theory turbulent diffusion and IPHS dissipative transport form.
        # Mathematical expression: sigma_h = grad(C_h).T star_K grad(C_h) >= 0.
        # Parameter meanings: star_K is the non-negative eddy diffusivity mobility.
        return {
            "dissipation_rate": concentration @ hodge @ concentration,
            "thermodynamic_force": concentration,
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for eddy diffusion."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive eddy diffusivity mobility."""

        n_dofs = len(self.target_species)
        hodge = self.build_hodge_contract(n_dofs=n_dofs, strict=True).matrix
        if bool(torch.any(hodge < 0.0)):
            raise ValueError("Eddy diffusion mobility must be non-negative in M_h.")
        return {
            "state_vars": tuple(
                self.species_map.get(species, f"{species}_mixing_ratio")
                for species in self.target_species
            ),
            "J": torch.zeros_like(hodge),
            "M": hodge,
            "ports": self.ports(),
            "open_system_route": "boundary_export",
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the boundary-layer exchange port.

        References:
          - Stull (1988), boundary-layer turbulent exchange.
          - Businger et al. (1971), flux-profile similarity.
        Mathematical expression:
          P_port = e_h q_h, q_h = -star_K D_h^T e_h.
        Parameter meanings:
          e_h is species mixing-ratio effort and q_h is vertical turbulent flux.
        """

        n_species = len(self.target_species)
        effort_space = CochainSpaceSpec(
            complex_id="atmosphere_boundary_layer_eddy_diffusion",
            degree=0,
            primality="primal",
            n_dofs=n_species,
            units="mol mol-1",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="boundary_layer_species_mixing_ratio",
            orientation_convention="upward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="atmosphere_boundary_layer_eddy_diffusion",
            degree=1,
            primality="dual",
            n_dofs=n_species,
            units="mol mol-1 m s-1",
            extensivity="integral",
            physical_domain="atmosphere",
            semantic_name="boundary_layer_turbulent_flux",
            orientation_convention="upward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="boundary_layer_exchange_port",
                effort_var="boundary_layer_species_mixing_ratio",
                flow_var="boundary_layer_turbulent_flux",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="mol2 mol-2 m s-1",
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
        self,
        state: StateDict,
        time: float,
        dt: float,
    ) -> FluxDict:
        """
        Forward pass for boundary layer eddy diffusion transport calculation.
        
        Compute boundary-layer mixing-ratio increments with the canonical
        K-theory relaxation closure.
        
        Parameters:
            state : StateDict
                Current system state with species mixing ratios [mol mol-1]
                Required: configured species mixing ratios, pbl_height,
                friction_velocity, and monin_obukhov_length.
            time : float
                Current simulation time [s]
            dt : float
                Time step [s]
        Returns:
            FluxDict
                Canonical state increments [mol mol-1] wrapped as
                UpdateItem(ACCUMULATE, per_timestep=True), plus diagnostics.
                
        Raises:
            ValueError: If required state variables missing
            RuntimeError: If transport calculation fails
            
        Mathematical Expression:
            Transport rate calculation:
            dC/dt = K/h^2 * (C_background - C)

            K [m2 s-1], h [m], and C [mol mol-1] give a mixing-ratio
            tendency [mol mol-1 s-1]. The returned increment is dC/dt * dt.
        """
        # References:
        # - Holton (2012), K-theory gradient transport.
        # - Stull (1988), boundary-layer eddy diffusion and explicit stability.
        # Mathematical expression:
        # - dC/dt = K / h^2 * (C_background - C).
        # - DeltaC = dt * dC/dt.
        # Parameter meanings:
        # - K is MOST-informed eddy diffusivity [m2 s-1].
        # - h is boundary-layer height [m].
        # - C and C_background are mixing ratios [mol mol-1].
        # Open-system route:
        # - C_background is the declared top-boundary forcing. The CFL bound
        #   makes the update a non-negative convex relaxation for C >= 0.
        # Early return if not enabled
        if not self.config_transport.enabled:
            return self._create_empty_output()
        
        try:
            # Validate required state variables
            self._validate_transport_state(state)

            dt_value = _to_float(dt)
            if dt_value <= 0.0:
                raise ValueError("Eddy diffusion dt must be positive.")

            # Get boundary layer height
            pbl_height = self._get_boundary_layer_height(state)
            
            # Get canonical turbulence state, produced by TurbulenceFunctor or
            # supplied as forcing under surface_layer_source = external_forcing.
            turbulence_data = self._get_turbulence_diagnostics(state)
            
            # Calculate eddy diffusivity with MOST corrections
            (
                eddy_diffusivity,
                stability_parameter,
                eddy_diffusivity_clamped,
                stability_parameter_clipping_residual,
            ) = self._calculate_eddy_diffusivity(
                pbl_height, turbulence_data
            )
            
            # Check CFL stability condition
            stability_number = self._check_cfl_stability(
                eddy_diffusivity, pbl_height, dt_value
            )
            
            # References: Stull (1988); Garratt (1992); Monin and Obukhov (1954).
            # Mathematical expression: dC/dt = K/h^2 * (C_background - C).
            # Parameter meanings: K is eddy diffusivity, C is mixing ratio, h is
            # boundary-layer height, and dt is the explicit transport time step.
            species_updates = self._calculate_species_transport(
                state, eddy_diffusivity, pbl_height, dt_value
            )
            
            # Store diagnostics
            self._update_diagnostics(
                eddy_diffusivity, pbl_height, turbulence_data,
                stability_number, stability_parameter, eddy_diffusivity_clamped,
                stability_parameter_clipping_residual,
            )
            
            # Combine updates with diagnostics
            output: FluxDict = {
                state_var: UpdateItem(
                    value=increment,
                    mode=UpdateMode.ACCUMULATE,
                    unit="mol mol-1",
                    per_timestep=True,
                    source_functor=self.name,
                )
                for state_var, increment in species_updates.items()
            }
            output.update(self._diagnostics)
            
            return output
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"Transport calculation failed in {self.name}: {str(e)}")
            raise RuntimeError(f"Eddy diffusion transport failed: {str(e)}") from e
    
    def _validate_transport_state(self, state: StateDict) -> None:
        """Validate state variables required for transport calculation."""
        required_vars = [
            'pbl_height',
            'friction_velocity',
            'monin_obukhov_length',
        ]
        for species_key in self.target_species:
            required_vars.append(
                self.species_map.get(species_key, f"{species_key}_mixing_ratio")
            )
        missing = [name for name in required_vars if name not in state]
        if missing:
            raise ValueError(
                "Missing required Eddy diffusion state variables: "
                + ", ".join(sorted(missing))
            )

        for name in required_vars:
            value = state[name]
            tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Eddy diffusion input '{name}' contains NaN or Inf.")

        for species_key in self.target_species:
            state_var = self.species_map.get(species_key, f"{species_key}_mixing_ratio")
            value = state[state_var]
            tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
            if bool(torch.any(tensor < 0.0)):
                raise ValueError(
                    f"Eddy diffusion input '{state_var}' must be non-negative."
                )
    
    def _get_boundary_layer_height(self, state: StateDict) -> Tensor:
        """
        Get the canonical boundary-layer height from state.
        
        Returns:
            Tensor: Validated boundary-layer height [m].
        """
        pbl_height = state['pbl_height']
        if not isinstance(pbl_height, Tensor):
            pbl_height = torch.tensor(float(pbl_height), device=self.device)
        else:
            pbl_height = pbl_height.to(device=self.device)
        pbl_min = self._transport_parameters["pbl_height_min"]
        pbl_max = self._transport_parameters["pbl_height_max"]
        if bool(torch.any((pbl_height < pbl_min) | (pbl_height > pbl_max))):
            raise ValueError(
                f"pbl_height must remain within [{pbl_min}, {pbl_max}] m."
            )
        return pbl_height
    
    def _get_turbulence_diagnostics(self,
                                   state: StateDict) -> Dict[str, Tensor]:
        """
        Extract canonical turbulence state from the selected surface-layer source.
        
        Returns:
            Dict[str, Tensor]: Turbulence parameters for MOST calculation
            Keys: 'friction_velocity', 'monin_obukhov_length'
        """
        friction_velocity = state['friction_velocity']
        if not isinstance(friction_velocity, Tensor):
            friction_velocity = torch.tensor(float(friction_velocity), device=self.device)
        else:
            friction_velocity = friction_velocity.to(device=self.device)
        if bool(
            torch.any(
                (friction_velocity < self._runtime_parameters["ustar_min"])
                | (friction_velocity > self._runtime_parameters["ustar_max"])
            )
        ):
            raise ValueError(
                "friction_velocity is outside configured Eddy diffusion bounds."
            )

        monin_obukhov_length = state['monin_obukhov_length']
        if not isinstance(monin_obukhov_length, Tensor):
            monin_obukhov_length = torch.as_tensor(
                monin_obukhov_length,
                device=self.device,
                dtype=torch.get_default_dtype(),
            )
        else:
            monin_obukhov_length = monin_obukhov_length.to(device=self.device)
            if not monin_obukhov_length.is_floating_point():
                monin_obukhov_length = monin_obukhov_length.to(
                    dtype=torch.get_default_dtype()
                )
        if bool(
            torch.any(
                torch.abs(monin_obukhov_length)
                <= torch.finfo(monin_obukhov_length.dtype).eps
            )
        ):
            raise ValueError("monin_obukhov_length must be non-zero.")

        return {
            'friction_velocity': friction_velocity,
            'monin_obukhov_length': monin_obukhov_length,
        }
    
    def _calculate_eddy_diffusivity(self,
                                  pbl_height: Tensor,
                                  turbulence_data: Dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calculate eddy diffusivity using MOST with stability corrections.
        
        References:
            Businger et al. (1971); Dyer (1974); Foken (2006).

        Mathematical Expression:
            zeta_raw = z/L
            l_s_eff = min(l_process, l_stable)
            l_u_eff = min(l_process, l_unstable)
            zeta_eff = clip(zeta_raw, -l_u_eff, l_s_eff)
            r_zeta = zeta_raw - zeta_eff
            K = kappau*z/phi_h(zeta_eff) where:
            - kappa = von Karman constant
            - u* = friction velocity
            - z = effective mixing height ~= h/2 (middle of BL)
            - phi_h = MOST stability function
            - L = Monin-Obukhov length
            - l_process = process-specific stability cap
            - l_stable and l_unstable = shared MOST operational limits
        
        Returns:
            Tuple containing eddy diffusivity [m2 s-1], the effective MOST
            stability parameter [dimensionless], a diffusivity clamp flag,
            and the stability-parameter clipping residual [dimensionless].
        """
        kappa = self._runtime_parameters["von_karman_constant"]
        u_star = turbulence_data['friction_velocity'] 
        L = turbulence_data['monin_obukhov_length']
        
        # Effective mixing height (middle of boundary layer)
        z_eff = pbl_height / 2.0
        
        # The shared MOST module owns the configured operational envelope.
        stable_limit = _to_float(
            self.get_parameter_value(
                "stable_correction_limit",
                namespace=ATMOS_MOST_NS,
                required=True,
            )
        )
        unstable_limit = _to_float(
            self.get_parameter_value(
                "unstable_correction_limit",
                namespace=ATMOS_MOST_NS,
                required=True,
            )
        )
        if (
            not math.isfinite(stable_limit)
            or stable_limit <= 0.0
            or not math.isfinite(unstable_limit)
            or unstable_limit <= 0.0
        ):
            raise ValueError("Shared MOST operational limits must be finite and positive.")

        process_limit = float(self._runtime_parameters["z_over_L_limit"])
        effective_stable_limit = min(process_limit, stable_limit)
        effective_unstable_limit = min(process_limit, unstable_limit)

        # Preserve the saturation route explicitly instead of silently discarding it.
        z_over_L_raw = z_eff / L
        z_over_L = torch.clamp(
            z_over_L_raw,
            min=-effective_unstable_limit,
            max=effective_stable_limit,
        )
        stability_parameter_clipping_residual = z_over_L_raw - z_over_L
        
        # MOST stability function phi_h (for scalars like temperature, species)
        phi_h = most_phi_h(z_over_L)
        
        # Calculate K using MOST
        K_raw = (kappa * u_star * z_eff) / phi_h
        
        # Apply stability bounds for numerical safety
        K = torch.clamp(
            K_raw,
            min=self._runtime_parameters["k_min"],
            max=self._runtime_parameters["k_max"],
        )
        clamped = torch.as_tensor(
            float(
                _to_bool((K_raw < self._runtime_parameters["k_min"]).any())
                or _to_bool((K_raw > self._runtime_parameters["k_max"]).any())
            ),
            device=K.device,
            dtype=K.dtype,
        )
        
        return K, z_over_L, clamped, stability_parameter_clipping_residual
    
    def _check_cfl_stability(self,
                            eddy_diffusivity: Tensor,
                            pbl_height: Tensor,
                            dt: float) -> Tensor:
        """
        Check CFL stability condition using unified operators.

        Returns:
            Tensor: Explicit diffusion stability number.

        Raises:
            ValueError: If the requested time step violates the stability limit.
        """
        # Use the minimum box height as a conservative characteristic length.
        characteristic_height = _to_float(pbl_height.min())

        # Reuse the side-effect-free canonical diffusion stability function.
        stability_number = compute_diffusion_stability(
            diffusivity=eddy_diffusivity,
            grid_spacing=characteristic_height,
            time_step=dt,
        )

        # Get maximum allowed stability number from parameters
        max_stability = self.get_parameter_value(
            "max_diffusion_stability",
            namespace=TRANSPORT_NUM_NS, required=True)
        allowed_stability = _to_float(max_stability) * float(
            self._runtime_parameters["cfl_factor"]
        )

        # Check stability
        max_stability_val = torch.max(stability_number) if isinstance(stability_number, torch.Tensor) else stability_number

        if bool(max_stability_val > allowed_stability):
            # Calculate suggested time step
            dt_suggested = dt * allowed_stability / _to_float(max_stability_val)
            raise ValueError(
                f"Diffusion stability condition violated in {self.name}: "
                f"stability={_to_float(max_stability_val):.6g} > "
                f"{allowed_stability:.6g}; requested dt={dt:.6g} s; "
                f"use dt <= {dt_suggested:.6g} s."
            )

        return stability_number if isinstance(stability_number, Tensor) else torch.as_tensor(
            stability_number,
            device=eddy_diffusivity.device,
            dtype=eddy_diffusivity.dtype,
        )
    
    def _calculate_species_transport(self,
                                   state: StateDict,
                                   eddy_diffusivity: Tensor,
                                   pbl_height: Tensor,
                                   dt: float) -> Dict[str, Tensor]:
        """
        Calculate transport rates for all configured species.
        
        Returns:
            Canonical state increments [mol mol-1].
        """
        updates: Dict[str, Tensor] = {}
        
        for species_key in self.target_species:
            state_var = self.species_map.get(species_key, f"{species_key}_mixing_ratio")
            
            # Get current mixing ratio
            mixing_ratio = state[state_var]
            if not isinstance(mixing_ratio, Tensor):
                mixing_ratio = torch.tensor(
                    mixing_ratio,
                    device=self.device,
                    dtype=torch.get_default_dtype(),
                )
            else:
                mixing_ratio = mixing_ratio.to(device=self.device)
                if not mixing_ratio.is_floating_point():
                    mixing_ratio = mixing_ratio.to(dtype=torch.get_default_dtype())
            
            # Calculate the canonical K-theory transport rate.
            transport_rate = self._calculate_single_species_transport(
                mixing_ratio, eddy_diffusivity, pbl_height, species_key
            )

            updates[state_var] = transport_rate * dt

        return updates
    
    def _calculate_single_species_transport(self,
                                          mixing_ratio: Tensor,
                                          eddy_diffusivity: Tensor,
                                          pbl_height: Tensor,
                                          species_key: str) -> Tensor:
        """
        Calculate transport rate for a single species.
        
        Mathematical Expression:
            r = K/h^2 * (C_background - C)
        """
        background_key = f"background_{species_key}_mixing_ratio"
        background_value = self._transport_parameters[background_key]
        background_ratio = torch.full_like(
            mixing_ratio,
            float(background_value),
            device=self.device,
        )
        return eddy_diffusivity / (pbl_height**2) * (
            background_ratio - mixing_ratio
        )
    
    def _update_diagnostics(self,
                           eddy_diffusivity: Tensor,
                           pbl_height: Tensor,
                           turbulence_data: Dict[str, Tensor],
                           stability_number: Tensor,
                           stability_parameter: Tensor,
                           eddy_diffusivity_clamped: Tensor,
                           stability_parameter_clipping_residual: Tensor) -> None:
        """Update transport diagnostics for output and monitoring."""
        self._diagnostics = {
            'diagnostics.eddy_diffusion.enabled': torch.ones(
                (), device=self.device
            ),
            'diagnostics.eddy_diffusion.eddy_diffusivity': eddy_diffusivity,
            'diagnostics.eddy_diffusion.pbl_height': pbl_height,
            'diagnostics.eddy_diffusion.friction_velocity': turbulence_data[
                'friction_velocity'
            ],
            'diagnostics.eddy_diffusion.monin_obukhov_length': turbulence_data[
                'monin_obukhov_length'
            ],
            'diagnostics.eddy_diffusion.stability_number': stability_number,
            'diagnostics.eddy_diffusion.stability_parameter_z_over_L': stability_parameter,
            'diagnostics.eddy_diffusion.eddy_diffusivity_clamped': eddy_diffusivity_clamped,
            'diagnostics.eddy_diffusion.stability_parameter_clipping_residual': (
                stability_parameter_clipping_residual
            ),
        }
    
    def _create_empty_output(self) -> FluxDict:
        """Create empty output when functor is disabled."""
        return {
            'diagnostics.eddy_diffusion.enabled': torch.zeros(
                (), device=self.device
            )
        }

    def _resolve_diffusivity_tensor(self, field_values: Tensor) -> Tensor:
        """Reject the inherited constant-diffusivity execution path."""
        raise RuntimeError(
            "Eddy diffusivity is state-dependent; use forward() with the "
            "declared pbl_height, friction_velocity, and "
            "monin_obukhov_length inputs."
        )

    def boundary_conditions(
        self,
        boundary_name: str,
        field_values: Tensor,
        normal_vector: Tensor,
        boundary_type: Optional[str] = None,
        boundary_value: Optional[float] = None,
        use_bc_manager: bool = True,
    ) -> Tensor:
        """Reject the generic boundary law that assumes constant diffusivity."""
        raise RuntimeError(
            "Eddy top-boundary exchange is evaluated only by the canonical "
            "forward() K-theory relaxation ledger."
        )

    def compute_diffusivity_tensor(
        self,
        temperature: Tensor,
        pressure: Tensor,
        medium_properties: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Reject the inherited molecular-diffusivity formula."""
        raise RuntimeError(
            "Eddy diffusivity requires PBL and turbulence state; use forward()."
        )
    
    def get_transport_metadata(self) -> Dict[str, Any]:
        """Get transport-specific metadata for introspection."""
        return {
            'transport_type': self.transport_type,
            'diffusivity_model': 'dynamic_most_k_theory',
            'surface_layer_source': self.surface_layer_source,
            'static_diffusivity': None,
            'activation_config': {
                'enabled': self.config_transport.enabled,
                'target_species': list(self.target_species),
            },
            'parameter_snapshot': {
                'von_karman_constant': self._runtime_parameters[
                    "von_karman_constant"
                ],
                'k_bounds': [
                    self._runtime_parameters["k_min"],
                    self._runtime_parameters["k_max"],
                ],
                'cfl_factor': self._runtime_parameters["cfl_factor"],
            },
            'species_mapping': self.species_map,
            'last_diagnostics': (
                self._diagnostics.copy() if self._diagnostics else None
            ),
        }
    
    def enable_transport(self, enabled: bool = True) -> None:
        """Enable or disable transport calculation."""
        self.config_transport.enabled = enabled
        logger.info(f"Transport {'enabled' if enabled else 'disabled'} for {self.name}")
    
    def __repr__(self) -> str:
        """Concise string representation."""
        enabled_str = "enabled" if self.config_transport.enabled else "disabled"
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"{enabled_str}, species={len(self.target_species)})")


# Export for module import
__all__ = ['EddyDiffusionBLMFunctor', 'EddyDiffusionConfig']
