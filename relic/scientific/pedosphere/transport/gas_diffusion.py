"""
Gas Diffusion Transport in Soil

This module implements one-dimensional vertical gas transport in soil systems
using Fick's law with effective diffusivity models. It handles diffusion of
CO2 and N2O gases through soil pores.

Scientific Basis:
----------------
The governing equation is:
J = -Deff partialC/partialz
F_surface,out = -J(z=0)

where:
- J = gas flux [g m^-2 s^-1]
- Deff = effective gas diffusion coefficient [m^2 s^-1]
- C = gas concentration [g m^-3]
- z = depth [m] (positive downward)
- F_surface,out = reported surface exchange [g element m^-2 s^-1]
  (positive from soil to atmosphere)

Effective diffusivity models:
1. Millington-Quirk: Deff = D0 * thetaa^(10/3) / phi^2
2. Penman: Deff = D0 * 0.66 * thetaa
3. DNDC: Special formulation for N2O with clay/porosity effects

where:
- D0 = diffusion coefficient in free air [m^2 s^-1]
- thetaa = air-filled porosity [m^3 m^-3]
- phi = total porosity [m^3 m^-3]

Temperature dependence:
D0(T) = D0(T0) * (T/T0)^n
where n ~= 1.75 for most gases

References:
----------
1. Millington, R.J. & Quirk, J.P. (1961). Permeability of porous solids. 
   Trans. Faraday Soc. 57, 1200-1207.
2. Penman, H.L. (1940). Gas and vapour movements in soil: I. The diffusion 
   of vapours through porous solids. J. Agric. Sci. 30, 437-462.
3. Moldrup, P. et al. (2000). Predicting the gas diffusion coefficient in 
   undisturbed soil from soil water characteristics. Soil Sci. Soc. Am. J. 
   64, 94-100.
4. Davidson, E.A. et al. (2012). The Dual Arrhenius and Michaelis-Menten 
   kinetics model for decomposition of soil organic matter at hourly to 
   seasonal time scales. Global Change Biology 18, 371-384.
5. Li, C. et al. (1992). A model of nitrous oxide evolution from soil driven 
   by rainfall events: 1. Model structure and sensitivity. J. Geophys. Res. 
   97, 9759-9776.
"""

import torch
import torch.nn.functional as F
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union
)
from dataclasses import dataclass
from enum import Enum
import logging

# Import unified transport operators for spatial calculations
from relic.computation.operators import OperatorFactory

# Import functions that are not yet available in computation layer
from relic.computation.operators import (
    harmonic_mean,
)
from relic.computation.operators.dec import CochainSpaceSpec, HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS

from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TransportMechanismFunctor,
    VariableKind,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.core.abstractions import StateDict, FluxDict
from relic.scientific.pedosphere.api_contract import (
    PedosphereContractMixin,
    standardize_output_keys,
    FluxSpecification,
    validate_functor_output,
    FluxDirection,
    FluxType,
)
from relic.utils.physical_constants import get_physical_constant

logger = logging.getLogger(__name__)


class DiffusivityModel(Enum):
    """Gas diffusivity models."""
    MILLINGTON_QUIRK = "millington_quirk"
    PENMAN = "penman"
    DNDC = "dndc"  # DNDC model formulation for N2O


@dataclass
class GasSpecies:
    """Configuration for a gas species."""
    name: str
    d0_at_stp: float  # Diffusivity in free air at STP [m^2 s^-1]
    temperature_exponent: float  # Temperature dependence exponent (no default - must be provided from parameter manager)
    atmospheric_concentration: Optional[float] = None  # Background concentration [g m^-3]


class GasDiffusion1D(PedosphereContractMixin, TransportMechanismFunctor):
    """
    One-dimensional vertical gas diffusion in soil.
    
    Implements Fick's law with effective diffusivity models for CO2 and N2O
    transport through soil air-filled pores. Supports multiple diffusivity
    models and temperature/moisture corrections. Fully integrated with
    ParameterManager and StateManager for consistent parameter access and
    state management.
    
    Features:
    - Millington-Quirk, Penman, and DNDC diffusivity models
    - Temperature-dependent diffusion coefficients
    - DNDC formulation for N2O with clay/porosity effects
    - Central difference numerical scheme
    - Atmospheric boundary condition
    - Full parameter management integration
    - Device-aware tensor operations
    - Unified dimension alignment
    - Conservation and stability controls
    """

    TRANSPORT_TYPE = "diffusion"
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'

    def __init__(self,
                 n_layers: Optional[int] = None,
                 soil_depth: Optional[float] = None,
                 diffusivity_model: Optional[str] = None,
                 enable_temperature_correction: Optional[bool] = None,
                 enable_atmospheric_exchange: Optional[bool] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize gas diffusion functor with full parameter management.
        
        Parameters:
        -----------
        n_layers : Optional[int]
            Number of soil layers (if None, loaded from ParameterManager)
        soil_depth : Optional[float]
            Total soil depth [m] (if None, loaded from ParameterManager)
        diffusivity_model : Optional[str]
            Model for effective diffusivity calculation
        enable_temperature_correction : Optional[bool]
            Apply temperature correction to diffusivity
        enable_atmospheric_exchange : Optional[bool]
            Enable exchange with atmosphere at surface
        device : Optional[torch.device]
            Computation device (CPU/CUDA)
        """
        # Set device
        self.device = device if device is not None else torch.device("cpu")
        
        # Initialize metadata
        metadata = FunctorMetadata(
            name="GasDiffusion1D",
            version="1.0.0",
            description="One-dimensional vertical gas diffusion in soil using Fick's law with effective diffusivity models",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="Jz = -Deff partialC/partialz; Fout = -Jz(0); deltaCi = -(Ji+1/2 - Ji-1/2) dt/dz",
            references=[
                "Millington, R.J. & Quirk, J.P. (1961). Trans. Faraday Soc. 57, 1200-1207",
                "Penman, H.L. (1940). J. Agric. Sci. 30, 437-462",
                "Moldrup, P. et al. (2000). Soil Sci. Soc. Am. J. 64, 94-100",
                "Li, C. et al. (1992). J. Geophys. Res. 97, 9759-9776",
                "Song, H. et al. (2021). GMD discussion supplement, Table S1, Eq. 20",
            ],
            parameters={
                "pedosphere.grid.n_layers": "Number of soil layers [dimensionless]",
                "pedosphere.grid.soil_depth": "Soil depth [m]",
                "pedosphere.transport.diffusion_model": "Effective diffusivity model id [dimensionless]",
                "pedosphere.transport.temperature_correction": "Enable temperature correction [dimensionless]",
                "pedosphere.transport.atmospheric_exchange": "Enable surface exchange [dimensionless]",
                "pedosphere.transport.allow_constructor_overrides": "Allow constructor overrides [dimensionless]",
                "pedosphere.transport.n2o_dndc_clay_effect": "DNDC N2O clay response coefficient [dimensionless]",
                "pedosphere.transport.n2o_dndc_porosity_effect": "DNDC N2O porosity response coefficient [dimensionless]",
                "pedosphere.transport.n2o_dndc_base_coefficient": "DNDC N2O base diffusion coefficient [dimensionless]",
                "pedosphere.transport.n2o_dndc_clay_porosity_coefficient": "DNDC clay-porosity coefficient [dimensionless]",
                "pedosphere.transport.n2o_dndc_temperature_base": "DNDC temperature response base [dimensionless]",
                "pedosphere.transport.n2o_dndc_temperature_divisor": "DNDC temperature response divisor [K]",
                "pedosphere.physical.co2_diffusivity_air_stp": "CO2 diffusivity in air at STP [m2 s-1]",
                "pedosphere.physical.n2o_diffusivity_air_stp": "N2O diffusivity in air at STP [m2 s-1]",
                "pedosphere.physical.gas_diffusivity_temperature_exponent": "Temperature exponent for gas diffusivity [dimensionless]",
                "pedosphere.physical.atmospheric_co2_concentration": "Atmospheric CO2 boundary concentration [g C m-3]",
                "pedosphere.physical.atmospheric_n2o_concentration": "Atmospheric N2O boundary concentration [g N m-3]",
                "pedosphere.physical.millington_quirk_exponent_numerator": "Millington-Quirk air-porosity exponent [dimensionless]",
                "pedosphere.physical.millington_quirk_exponent_denominator": "Millington-Quirk total-porosity exponent [dimensionless]",
                "pedosphere.physical.penman_coefficient": "Penman gas tortuosity coefficient [dimensionless]",
                "pedosphere.numerical.clamp_min": "Minimum admissible numerical value [varies]",
                "pedosphere.numerical.epsilon": "Dimensionless numerical tolerance [dimensionless]",
                "pedosphere.numerical.epsilon_length": "Minimum positive length [m]",
                "pedosphere.numerical.min_time_step": "Minimum stable time step [s]",
                "pedosphere.numerical.zero_initial": "Canonical numerical zero [dimensionless]",
                "pedosphere.numerical.wfps_minimum": "Minimum water-filled pore-space fraction [dimensionless]",
                "pedosphere.numerical.wfps_maximum": "Maximum water-filled pore-space fraction [dimensionless]",
            },
            input_variables=["soil_moisture", "porosity", "soil_temperature", "clay_content",
                           "soil_co2_concentration", "n2o_concentration"],
            output_variables=[
                "co2_surface_flux",
                "n2o_surface_flux",
                "co2_flux_profile",
                "n2o_flux_profile",
                "soil_co2_concentration",
                "n2o_concentration",
                "diagnostics.gas_diffusion.energy_storage",
                "diagnostics.gas_diffusion.dissipation_rate",
                "diagnostics.gas_diffusion.dH_dt_audit",
                "diagnostics.gas_diffusion.port_power_residual",
                "diagnostics.gas_diffusion.co2_clipping_residual",
                "diagnostics.gas_diffusion.n2o_clipping_residual",
                "diagnostics.gas_diffusion.step_seconds",
            ],
            conservation_laws=["mass"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.STATE,
            spatial_dimensions=1
        )
        
        # Create configuration dictionary
        parameter_config = {
            "n_layers": n_layers,
            "soil_depth": soil_depth,
            "diffusivity_model": diffusivity_model,
            "enable_temperature_correction": enable_temperature_correction,
            "enable_atmospheric_exchange": enable_atmospheric_exchange,
        }
        config = {"parameters": {k: v for k, v in parameter_config.items() if v is not None}}
        
        # Initialize parent class
        super().__init__(name="GasDiffusion1D", config=config, metadata=metadata)

        self.reference_temperature = float(
            get_physical_constant("reference_temperature")
        )
        self.celsius_offset = float(get_physical_constant("celsius_to_kelvin"))
        self.seconds_per_hour = float(get_physical_constant("seconds_per_hour"))
        
        # Load configuration from ParameterManager
        self._load_config_from_parameter_manager()
        
        # Setup vertical grid
        self._setup_vertical_grid()
        
        # Configure gas species
        self._setup_species_config()
        
        logger.info(f"Initialized {self.metadata.name} with {self.diffusivity_model} model")

    def _load_config_from_parameter_manager(self) -> None:
        """Load configuration from unified parameter manager."""
        override_params = {}
        if isinstance(self._original_config, dict):
            override_params = self._original_config.get("parameters", {}) or {}

        allow_overrides = self.get_parameter_value(
            "allow_constructor_overrides",
            namespace="pedosphere.transport", required=True)
        if isinstance(allow_overrides, torch.Tensor):
            allow_overrides = bool(allow_overrides.detach())
        else:
            allow_overrides = bool(allow_overrides)

        self._constructor_overrides_used = []
        self._constructor_overrides_blocked = []

        def _resolve_override(key: str, full_key: str) -> Any:
            if key in override_params:
                if allow_overrides:
                    self._constructor_overrides_used.append(full_key)
                    return override_params[key]
                self._constructor_overrides_blocked.append(full_key)
            return None

        # Grid configuration
        n_layers_override = _resolve_override("n_layers", "pedosphere.grid.n_layers")
        if n_layers_override is None:
            self.n_layers = self.get_parameter_value(
                "n_layers",
                namespace="pedosphere.grid", required=True)
        else:
            self.n_layers = n_layers_override
        if isinstance(self.n_layers, torch.Tensor):
            self.n_layers = int(self.n_layers.detach())
        else:
            self.n_layers = int(self.n_layers)

        soil_depth_override = _resolve_override("soil_depth", "pedosphere.grid.soil_depth")
        if soil_depth_override is None:
            self.soil_depth = self.get_parameter_value(
                "soil_depth",
                namespace="pedosphere.grid", required=True)
        else:
            self.soil_depth = soil_depth_override

        # Gas diffusion configuration
        model_override = _resolve_override(
            "diffusivity_model", "pedosphere.transport.diffusion_model"
        )
        if model_override is None:
            model_str = self.get_parameter_value(
                "diffusion_model",
                namespace="pedosphere.transport", required=True)
        else:
            model_str = model_override
        self.diffusivity_model = (
            model_str if isinstance(model_str, DiffusivityModel) else DiffusivityModel(model_str)
        )

        temp_override = _resolve_override(
            "enable_temperature_correction", "pedosphere.transport.temperature_correction"
        )
        if temp_override is None:
            temp_correction = self.get_parameter_value(
                "temperature_correction",
                namespace="pedosphere.transport", required=True)
        else:
            temp_correction = temp_override
        if isinstance(temp_correction, torch.Tensor):
            self.enable_temperature_correction = bool(temp_correction.detach())
        else:
            self.enable_temperature_correction = bool(temp_correction)

        atm_override = _resolve_override(
            "enable_atmospheric_exchange", "pedosphere.transport.atmospheric_exchange"
        )
        if atm_override is None:
            atm_exchange = self.get_parameter_value(
                "atmospheric_exchange",
                namespace="pedosphere.transport", required=True)
        else:
            atm_exchange = atm_override
        if isinstance(atm_exchange, torch.Tensor):
            self.enable_atmospheric_exchange = bool(atm_exchange.detach())
        else:
            self.enable_atmospheric_exchange = bool(atm_exchange)

        if self._constructor_overrides_blocked:
            logger.warning(
                "Constructor overrides ignored for %s; set pedosphere.transport.allow_constructor_overrides to enable.",
                ", ".join(self._constructor_overrides_blocked),
            )
    
    def _setup_vertical_grid(self) -> None:
        """Setup vertical grid as module buffers for device consistency."""
        # Calculate layer properties
        default_dtype = torch.get_default_dtype()
        layer_thickness = torch.full(
            (self.n_layers,),
            float(self.soil_depth) / self.n_layers,
            dtype=default_dtype,
            device=self.device,
        )
        
        layer_centers = torch.arange(
            self.n_layers,
            dtype=default_dtype,
            device=self.device,
        ) * (float(self.soil_depth) / self.n_layers) + (float(self.soil_depth) / self.n_layers / 2)
        
        # Register as buffers for automatic device migration
        self.register_buffer('layer_thickness', layer_thickness)
        self.register_buffer('layer_centers', layer_centers)
    
    def _setup_species_config(self) -> None:
        """Setup gas species configuration from parameter manager."""
        self.species = {}
        
        # CO2 configuration
        self.species["CO2"] = GasSpecies(
            name="CO2",
            d0_at_stp=self.get_parameter_value(
                "co2_diffusivity_air_stp",
                namespace="pedosphere.physical", required=True),
            temperature_exponent=self.get_parameter_value(
                "gas_diffusivity_temperature_exponent",
                namespace="pedosphere.physical", required=True),
            atmospheric_concentration=self.get_parameter_value(
                "atmospheric_co2_concentration",
                namespace="pedosphere.physical", required=True)
        )
        
        # N2O configuration
        self.species["N2O"] = GasSpecies(
            name="N2O",
            d0_at_stp=self.get_parameter_value(
                "n2o_diffusivity_air_stp",
                namespace="pedosphere.physical", required=True),
            temperature_exponent=self.get_parameter_value(
                "gas_diffusivity_temperature_exponent",
                namespace="pedosphere.physical", required=True),
            atmospheric_concentration=self.get_parameter_value(
                "atmospheric_n2o_concentration",
                namespace="pedosphere.physical", required=True)
        )
        
    def get_dependencies(self) -> Dict[str, List[str]]:
        """Get required input dependencies."""
        return self._merge_declared_dependencies({
            "required": [],
            "optional": [
                "first_order_decomposition",
                "microbial_enzyme_decomposition",
                "nitrification",
                "denitrification"
            ]
        })
    
    def get_output_specifications(self) -> Dict[str, FluxSpecification]:
        """Get output variable specifications."""
        return {
            "co2_surface_flux": FluxSpecification(
                name="co2_surface_flux",
                units="g C m-2 s-1",
                direction=FluxDirection.OUTPUT,
                flux_type=FluxType.EMISSION,
                description="CO2 surface exchange, positive from soil to atmosphere",
                is_rate=True
            ),
            "n2o_surface_flux": FluxSpecification(
                name="n2o_surface_flux",
                units="g N m-2 s-1",
                direction=FluxDirection.OUTPUT,
                flux_type=FluxType.EMISSION,
                description="N2O surface exchange, positive from soil to atmosphere",
                is_rate=True
            ),
            "co2_flux_profile": FluxSpecification(
                name="co2_flux_profile",
                units="g C m-2 s-1",
                direction=FluxDirection.OUTPUT,
                flux_type=FluxType.EMISSION,
                description="CO2 flux profile between layers",
                is_rate=True,
            ),
            "n2o_flux_profile": FluxSpecification(
                name="n2o_flux_profile",
                units="g N m-2 s-1",
                direction=FluxDirection.OUTPUT,
                flux_type=FluxType.EMISSION,
                description="N2O flux profile between layers",
                is_rate=True,
            ),
        }

    def _structural_n_dofs(self, n_dofs: Optional[int] = None) -> int:
        """Return structural audit degrees of freedom for two gas species."""

        value = int(n_dofs) if n_dofs is not None else int(self.n_layers)
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for soil gas diffusion.

        References:
          - Millington and Quirk (1961), porous-medium gas diffusion.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          c_dot = -D_h q_h + G_h u_h, q_h = -star_K d_h c_h.
        Parameter meanings:
          n_layers defines the local cochain support for each gas species.
        """

        n_layers = self._structural_n_dofs()
        n_species_dofs = 2 * n_layers
        return (
            CochainSpaceSpec(
                complex_id="pedosphere_soil_gas_diffusion",
                degree=0,
                primality="primal",
                n_dofs=n_layers,
                units="g C m-3",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="soil_co2_concentration",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_soil_gas_diffusion",
                degree=0,
                primality="primal",
                n_dofs=n_layers,
                units="g N m-3",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="n2o_concentration",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_soil_gas_diffusion",
                degree=1,
                primality="primal",
                n_dofs=n_species_dofs,
                units="g m-4",
                extensivity="intensive",
                physical_domain="pedosphere",
                semantic_name="soil_gas_concentration_gradient",
                orientation_convention="outward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="pedosphere_soil_gas_diffusion",
                degree=2,
                primality="dual",
                n_dofs=n_species_dofs,
                units="g m-2 s-1",
                extensivity="integral",
                physical_domain="pedosphere",
                semantic_name="soil_gas_diffusive_flux",
                orientation_convention="outward_positive",
                boundary_role="boundary",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "pedosphere.grid.n_layers",
            "pedosphere.grid.soil_depth",
            "pedosphere.transport.diffusion_model",
            "pedosphere.transport.temperature_correction",
            "pedosphere.transport.atmospheric_exchange",
            "pedosphere.transport.allow_constructor_overrides",
            "pedosphere.transport.n2o_dndc_clay_effect",
            "pedosphere.transport.n2o_dndc_porosity_effect",
            "pedosphere.transport.n2o_dndc_base_coefficient",
            "pedosphere.transport.n2o_dndc_clay_porosity_coefficient",
            "pedosphere.transport.n2o_dndc_temperature_base",
            "pedosphere.transport.n2o_dndc_temperature_divisor",
            "pedosphere.physical.co2_diffusivity_air_stp",
            "pedosphere.physical.n2o_diffusivity_air_stp",
            "pedosphere.physical.gas_diffusivity_temperature_exponent",
            "pedosphere.physical.atmospheric_co2_concentration",
            "pedosphere.physical.atmospheric_n2o_concentration",
            "pedosphere.physical.millington_quirk_exponent_numerator",
            "pedosphere.physical.millington_quirk_exponent_denominator",
            "pedosphere.physical.penman_coefficient",
            "pedosphere.numerical.clamp_min",
            "pedosphere.numerical.epsilon",
            "pedosphere.numerical.epsilon_length",
            "pedosphere.numerical.min_time_step",
            "pedosphere.numerical.zero_initial",
            "pedosphere.numerical.wfps_minimum",
            "pedosphere.numerical.wfps_maximum",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the soil-gas diffusion Hodge contract.

        References:
          - Millington and Quirk (1961), effective gas diffusivity.
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          q_h = -star_K d_h c_h with star_K = diag(D_eff) and D_eff >= 0.
        Parameter meanings:
          CO2 and N2O free-air diffusivities initialize the passive audit Hodge.
          n_dofs is the structural support per gas species.
        """

        n_layers = self._structural_n_dofs(n_dofs=n_dofs)
        co2_d0 = self._ensure_tensor(self.species["CO2"].d0_at_stp).reshape(1)
        n2o_d0 = self._ensure_tensor(self.species["N2O"].d0_at_stp).reshape(1)
        values = torch.cat(
            [
                co2_d0.expand(n_layers),
                n2o_d0.expand(n_layers),
            ],
            dim=0,
        ).to(device=self.device, dtype=torch.float32)
        if bool(torch.any(values < 0.0)):
            raise ValueError("Gas diffusivity must be non-negative for Hodge assembly.")

        domain = CochainSpaceSpec(
            complex_id="pedosphere_soil_gas_diffusion",
            degree=1,
            primality="primal",
            n_dofs=2 * n_layers,
            units="g m-4",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="soil_gas_concentration_gradient",
            orientation_convention="outward_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="pedosphere_soil_gas_diffusion",
            degree=2,
            primality="dual",
            n_dofs=2 * n_layers,
            units="g m-2 s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="soil_gas_diffusive_flux",
            orientation_convention="outward_positive",
            boundary_role="boundary",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=(
                "pedosphere.physical.co2_diffusivity_air_stp",
                "pedosphere.physical.n2o_diffusivity_air_stp",
            ),
            metric_keys=("pedosphere.grid.soil_depth", "pedosphere.grid.n_layers"),
            material_law="soil_gas_fickian_diffusion",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build soil-gas Hodge terms for Phase 8 IPHS assembly."""

        n_dofs = getattr(complex_bundle, "n_dofs", None) if complex_bundle is not None else None
        if n_dofs is not None:
            n_dofs = max(1, int(n_dofs) // 2)
        return (self.build_hodge_contract(n_dofs=n_dofs, strict=True),)

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return storage functional and effort terms for strict transport audits."""

        if not self._has_gas_concentration_state(state):
            return {
                "storage_functional": "H_h = 0.5 * c_h.T c_h",
                "effort": "e_h = delta H_h / delta c_h = c_h",
            }
        concentration = self._gas_concentration_vector(state)
        return {
            "storage_functional": self._quadratic_gas_storage(concentration),
            "effort": concentration,
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return non-negative soil-gas diffusion dissipation terms."""

        if not self._has_gas_transport_state(state):
            return {
                "dissipation_rate": "sigma_h = X_h.T star_K X_h",
                "thermodynamic_force": "X_h = d_h c_h",
            }
        terms = self._gas_transport_terms(state)
        return {
            "dissipation_rate": self._gas_diffusion_dissipation_rate(
                terms["co2_conc"],
                terms["n2o_conc"],
                terms["co2_deff"],
                terms["n2o_deff"],
            ),
            "thermodynamic_force": "X_h = d_h c_h",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for pure soil gas diffusion."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive soil-gas diffusion mobility contribution.

        References:
          - Millington and Quirk (1961), porous-medium effective diffusivity.
          - van der Schaft and Maschke (1995), dissipative port-Hamiltonian form.
        Mathematical expression:
          M_h = diag(D_eff), D_eff >= 0, sigma = X_h.T M_h X_h >= 0.
        Parameter meanings:
          D_eff is assembled from ParameterManager-backed diffusivity, porosity,
          moisture, temperature, and DNDC boundary-coefficient settings.
        """

        if self._has_gas_transport_state(state):
            terms = self._gas_transport_terms(state)
            values = torch.cat(
                [terms["co2_deff"].reshape(-1), terms["n2o_deff"].reshape(-1)],
                dim=0,
            ).to(device=self.device, dtype=torch.float32)
        else:
            values = torch.diag(self.build_hodge_contract(strict=True).matrix)
        if bool(torch.any(values < 0.0)):
            raise ValueError("Soil gas mobility must be non-negative in M_h.")
        matrix = torch.diag(values)
        n_layers = int(values.numel() // 2)
        return {
            "state_vars": tuple(
                [f"soil_co2_concentration[{idx}]" for idx in range(n_layers)]
                + [f"n2o_concentration[{idx}]" for idx in range(n_layers)]
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare soil-atmosphere gas boundary ports.

        References:
          - Penman (1940), gas exchange at soil-atmosphere boundary.
          - van der Schaft and Maschke (1995), port power pairing.
        Mathematical expression:
          P_port = e_h * f_h with e_h = c_surface and f_h = J_surface.
        Parameter meanings:
          Atmospheric concentrations define the boundary efforts through YAML.
        """

        co2_effort_space = CochainSpaceSpec(
            complex_id="pedosphere_soil_gas_diffusion",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="g C m-3",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="soil_atmosphere_co2_effort",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        n2o_effort_space = CochainSpaceSpec(
            complex_id="pedosphere_soil_gas_diffusion",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="g N m-3",
            extensivity="intensive",
            physical_domain="pedosphere",
            semantic_name="soil_atmosphere_n2o_effort",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        co2_flow_space = CochainSpaceSpec(
            complex_id="pedosphere_soil_gas_diffusion",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="g C m-2 s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="soil_atmosphere_co2_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        n2o_flow_space = CochainSpaceSpec(
            complex_id="pedosphere_soil_gas_diffusion",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="g N m-2 s-1",
            extensivity="integral",
            physical_domain="pedosphere",
            semantic_name="soil_atmosphere_n2o_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="soil_atmosphere_co2_flux",
                effort_var="soil_co2_surface_concentration",
                flow_var="co2_surface_flux",
                effort_space=co2_effort_space,
                flow_space=co2_flow_space,
                power_units="g2 C2 m-5 s-1",
                positive_power_convention="out_of_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="soil_atmosphere_n2o_flux",
                effort_var="soil_n2o_surface_concentration",
                flow_var="n2o_surface_flux",
                effort_space=n2o_effort_space,
                flow_space=n2o_flow_space,
                power_units="g2 N2 m-5 s-1",
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
    
    def _ensure_tensor(self, value: Union[float, torch.Tensor], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Ensure value is a tensor with proper device and dtype.
        
        Parameters:
        -----------
        value : Union[float, torch.Tensor]
            Value to convert
        dtype : Optional[torch.dtype]
            Target dtype (default: float32)
            
        Returns:
        --------
        torch.Tensor
            Properly formatted tensor on correct device
        """
        if isinstance(value, torch.Tensor):
            tensor = value
        else:
            dtype = dtype or torch.float32
            tensor = torch.tensor(float(value), dtype=dtype, device=self.device)
        
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        
        return tensor
    
    def _align_profile(self, tensor: torch.Tensor, target_layers: Optional[int] = None) -> torch.Tensor:
        """
        Align profile tensor to target number of layers.
        
        Implements unified dimension alignment strategy:
        - Scalar: broadcast to all layers
        - Matching dimensions: return as-is
        - Mismatched dimensions: resample
        
        Parameters:
        -----------
        tensor : torch.Tensor
            Input tensor to align
        target_layers : Optional[int]
            Target number of layers (default: self.n_layers)
            
        Returns:
        --------
        torch.Tensor
            Aligned tensor with shape (target_layers,)
        """
        if target_layers is None:
            target_layers = self.n_layers
        
        # Ensure tensor is on correct device
        tensor = self._ensure_tensor(tensor)
        
        # Scalar -> broadcast
        if tensor.numel() == 1:
            return tensor.expand(target_layers)
        
        # Dimension matches -> return reshaped
        elif tensor.numel() == target_layers:
            return tensor.view(target_layers)
        
        # Dimension mismatch -> resample
        else:
            return self._resample_profile(tensor, target_layers)
    
    def _resample_profile(self, tensor: torch.Tensor, target_layers: int) -> torch.Tensor:
        """
        Resample profile to target number of layers using linear interpolation.
        
        Parameters:
        -----------
        tensor : torch.Tensor
            Input profile tensor
        target_layers : int
            Target number of layers
            
        Returns:
        --------
        torch.Tensor
            Resampled profile
        """
        source_layers = tensor.numel()

        # References: LeVeque (2007) Finite Difference Methods; Strikwerda (2004) FD Schemes.
        # Mathematical: C_interp = interp(C); scale = sum(C) / sum(C_interp); C_resampled = C_interp * scale
        # Parameters: C [g m-2 s-1], target_layers [-], epsilon [dimensionless]
        source = tensor.reshape(1, 1, source_layers)
        resampled = F.interpolate(
            source,
            size=target_layers,
            mode="linear",
            align_corners=False,
        ).reshape(-1)

        total_before = tensor.sum()
        total_after = resampled.sum()
        epsilon = float(
            self.get_parameter_value(
                "epsilon", namespace="pedosphere.numerical", required=True
            )
        )
        epsilon_tensor = torch.tensor(
            float(epsilon),
            device=resampled.device,
            dtype=resampled.dtype,
        )
        if torch.abs(total_after) > epsilon_tensor:
            resampled = resampled * (total_before / total_after)

        return resampled

    def validate_inputs(self, state: StateDict) -> List[str]:
        """
        Validate input state variables.
        
        Parameters:
        -----------
        state : StateDict
            Current state dictionary
            
        Returns:
        --------
        List[str]
            List of validation errors (empty if valid)
        """
        errors = []
        required = self.get_required_inputs()
        
        for var_name in required:
            if var_name not in state:
                errors.append(f"Missing required input: {var_name}")
                continue

            value = state[var_name]
            if value is None:
                errors.append(f"Missing required input value for: {var_name}")
                continue
            if not isinstance(value, torch.Tensor):
                errors.append(f"{var_name} must be torch.Tensor, got {type(value)}")
                continue

            if not bool(torch.all(torch.isfinite(value))):
                errors.append(f"{var_name} must be finite")
                continue
            
            # Check for physical ranges
            if var_name in ["soil_moisture", "porosity", "clay_content"]:
                if torch.any(value < 0) or torch.any(value > 1):
                    errors.append(f"{var_name} must be in [0,1]")
            
            elif var_name == "soil_temperature":
                if torch.any(value < 200) or torch.any(value > 350):
                    errors.append("soil_temperature out of range [200, 350] K")
            
            elif var_name in ["soil_co2_concentration", "n2o_concentration"]:
                if value.numel() != self.n_layers:
                    errors.append(
                        f"{var_name} must have {self.n_layers} layers, "
                        f"got {value.numel()}"
                    )
                if torch.any(value < 0):
                    errors.append(f"{var_name} must be non-negative")

        moisture = state.get("soil_moisture")
        porosity = state.get("porosity")
        if isinstance(moisture, torch.Tensor) and isinstance(porosity, torch.Tensor):
            if bool(torch.all(torch.isfinite(moisture))) and bool(
                torch.all(torch.isfinite(porosity))
            ):
                try:
                    moisture_profile = self._align_profile(moisture)
                    porosity_profile = self._align_profile(porosity)
                    if bool(torch.any(moisture_profile > porosity_profile)):
                        errors.append("soil_moisture must not exceed porosity")
                except (RuntimeError, ValueError) as exc:
                    errors.append(f"soil_moisture and porosity shape mismatch: {exc}")
        
        return errors

    def _has_gas_concentration_state(self, state: Mapping[str, Any]) -> bool:
        """Return whether CO2 and N2O concentration cochains are available."""

        return (
            isinstance(state, Mapping)
            and "soil_co2_concentration" in state
            and "n2o_concentration" in state
            and isinstance(state.get("soil_co2_concentration"), torch.Tensor)
            and isinstance(state.get("n2o_concentration"), torch.Tensor)
        )

    def _has_gas_transport_state(self, state: Mapping[str, Any]) -> bool:
        """Return whether state contains the fields needed for gas Hodge assembly."""

        if not self._has_gas_concentration_state(state):
            return False
        return all(
            key in state
            for key in (
                "soil_moisture",
                "porosity",
                "soil_temperature",
                "clay_content",
            )
        )

    def _gas_concentration_vector(self, state: Mapping[str, Any]) -> torch.Tensor:
        """Return the stacked CO2 and N2O concentration cochain."""

        co2_conc = self._align_profile(state["soil_co2_concentration"])
        n2o_conc = self._align_profile(state["n2o_concentration"])
        return torch.cat([co2_conc.reshape(-1), n2o_conc.reshape(-1)], dim=0)

    def _quadratic_gas_storage(self, concentration: torch.Tensor) -> torch.Tensor:
        """Compute the quadratic concentration storage functional.

        References:
          - Crank (1975), diffusion gradient-flow energy estimates.
          - van der Schaft and Maschke (1995), port-Hamiltonian storage.
        Mathematical expression:
          H_h = 0.5 * c_h.T c_h.
        Parameter meanings:
          c_h contains stacked CO2 and N2O layer concentration cochains.
        """

        return 0.5 * torch.sum(concentration.reshape(-1) * concentration.reshape(-1))

    def _gas_transport_terms(self, state: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Assemble current gas concentrations and effective diffusivities."""

        working_state = dict(state)
        theta = self._align_profile(working_state["soil_moisture"])
        porosity = self._align_profile(working_state["porosity"])
        temperature = self._align_profile(working_state["soil_temperature"])
        clay = self._align_profile(working_state["clay_content"])
        co2_conc = self._align_profile(working_state["soil_co2_concentration"])
        n2o_conc = self._align_profile(working_state["n2o_concentration"])
        clamp_min = float(
            self.get_parameter_value(
                "clamp_min", namespace="pedosphere.numerical", required=True
            )
        )
        air_porosity = torch.clamp(porosity - theta, min=clamp_min)
        co2_deff = self._calculate_effective_diffusivity(
            air_porosity, porosity, temperature, clay, "CO2"
        )
        n2o_deff = self._calculate_effective_diffusivity(
            air_porosity, porosity, temperature, clay, "N2O"
        )
        return {
            "co2_conc": co2_conc,
            "n2o_conc": n2o_conc,
            "co2_deff": co2_deff,
            "n2o_deff": n2o_deff,
        }

    def _interface_dissipation(
        self,
        concentration: torch.Tensor,
        deff: torch.Tensor,
    ) -> torch.Tensor:
        """Compute non-negative diffusion dissipation over layer interfaces.

        References:
          - Millington and Quirk (1961), effective diffusivity in porous media.
          - Crank (1975), Fickian diffusion dissipation.
        Mathematical expression:
          sigma_h = sum_i D_eff,i+1/2 * ((c_i+1 - c_i) / dz)^2.
        Parameter meanings:
          concentration: layer gas concentration [g m-3].
          deff: layer effective gas diffusivity [m2 s-1].
          dz: layer spacing from the ParameterManager-backed grid.
        """

        if concentration.numel() < 2:
            zero_value = float(
                self.get_parameter_value(
                    "zero_initial", namespace="pedosphere.numerical", required=True
                )
            )
            return torch.as_tensor(zero_value, device=self.device, dtype=concentration.dtype)
        dz_min = float(
            self.get_parameter_value(
                "epsilon_length", namespace="pedosphere.numerical", required=True
            )
        )
        dz = torch.as_tensor(
            max(float(self.soil_depth) / float(self.n_layers), dz_min),
            device=self.device,
            dtype=concentration.dtype,
        )
        gradient = (concentration[1:] - concentration[:-1]) / dz
        interface_deff = harmonic_mean(deff[:-1], deff[1:])
        return torch.sum(interface_deff * gradient * gradient)

    def _gas_diffusion_dissipation_rate(
        self,
        co2_conc: torch.Tensor,
        n2o_conc: torch.Tensor,
        co2_deff: torch.Tensor,
        n2o_deff: torch.Tensor,
    ) -> torch.Tensor:
        """Return the total non-negative CO2 plus N2O diffusion dissipation."""

        return self._interface_dissipation(
            co2_conc, co2_deff
        ) + self._interface_dissipation(
            n2o_conc, n2o_deff
        )

    def _iphs_audit_terms(
        self,
        co2_before: torch.Tensor,
        n2o_before: torch.Tensor,
        co2_delta: torch.Tensor,
        n2o_delta: torch.Tensor,
        co2_deff: torch.Tensor,
        n2o_deff: torch.Tensor,
        co2_surface_flux: torch.Tensor,
        n2o_surface_flux: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> Dict[str, torch.Tensor]:
        """Compute IPHS-compatible storage, dissipation, and port-power diagnostics."""

        # References:
        #   - Crank (1975), diffusion energy balance.
        #   - Millington and Quirk (1961), soil gas effective diffusivity.
        #   - van der Schaft and Maschke (1995), port-Hamiltonian systems.
        # Mathematical expression:
        #   dH/dt = -sigma_h + e_h.T G_h u_h + P_residual.
        #   sigma_h = X_h.T star_K X_h >= 0.
        # Parameter meanings:
        #   c_h stores CO2 and N2O concentration cochains.
        #   u_h stores soil-atmosphere boundary gas flux ports.
        dt_safe = torch.as_tensor(
            dt,
            device=self.device,
            dtype=co2_before.dtype,
        )
        min_dt = float(
            self.get_parameter_value(
                "min_time_step", namespace="pedosphere.numerical", required=True
            )
        )
        dt_safe = torch.clamp(dt_safe, min=min_dt)
        before = torch.cat([co2_before.reshape(-1), n2o_before.reshape(-1)], dim=0)
        after = torch.cat(
            [
                (co2_before + co2_delta).reshape(-1),
                (n2o_before + n2o_delta).reshape(-1),
            ],
            dim=0,
        )
        storage_before = self._quadratic_gas_storage(before)
        storage_after = self._quadratic_gas_storage(after)
        dH_dt = (storage_after - storage_before) / dt_safe
        dissipation = self._gas_diffusion_dissipation_rate(
            co2_before,
            n2o_before,
            co2_deff,
            n2o_deff,
        )
        port_power = (
            co2_before.reshape(-1)[0] * co2_surface_flux.reshape(-1)[0]
            + n2o_before.reshape(-1)[0] * n2o_surface_flux.reshape(-1)[0]
        )
        port_power_residual = dH_dt + dissipation - port_power
        return {
            "energy_storage": storage_after,
            "dissipation_rate": dissipation,
            "dH_dt_audit": dH_dt,
            "port_power_residual": port_power_residual,
        }
    
    def forward(self,
        # References: Millington and Quirk (1961); Penman (1940); Li et al. (1992), DNDC;
        #   LeVeque (2002).
        # Mathematical expression: D_eff by the selected model (Millington-Quirk, Penman or the
        #   DNDC factor); J = -D_eff dC/dz at layer interfaces; F_out = -J_top (Fickian) or the
        #   DNDC surface expression, both zero when atmospheric exchange is disabled;
        #   Delta C_i = -(J_{i+1/2} - J_{i-1/2}) dt / dz bounded at zero with the signed clipping routed.
        # Parameter meanings: theta soil water, phi porosity, phi - theta air-filled porosity,
        #   T soil temperature, clay fraction, C species concentration, C_atm the atmospheric
        #   boundary concentration.
        # Units: C [g C m-3] or [g N m-3]; D_eff [m2 s-1]; J, F_out [g m-2 s-1]; dz [m];
        #   dt [s]; clipping residuals [g m-3].
        # Domain/constraints: profiles hold exactly n_layers entries; no-flow bottom boundary;
        #   the top boundary is closed for both species when enable_atmospheric_exchange is false.
        # Sign convention: F_out positive from soil to atmosphere; J positive downward.
                state: StateDict,
                time: float,
                dt: torch.Tensor | float,
                **kwargs) -> FluxDict:
        """
        Execute gas diffusion for one time step.
        
        Parameters:
        -----------
        state : StateDict
            Current system state
        time : float
            Current simulation time [s]
        dt : float
            Time step size [s]
        **kwargs
            Additional parameters
        
        Returns:
        --------
        FluxDict
            Gas fluxes and surface emissions
        """
        working_state = dict(state)

        # Validate inputs
        errors = self.validate_inputs(working_state)
        if errors:
            raise ValueError(f"Input validation failed: {errors}")

        dt_tensor = torch.as_tensor(dt, device=self.device)
        if dt_tensor.numel() != 1:
            raise ValueError("dt must be a scalar")
        if not bool(torch.all(torch.isfinite(dt_tensor))):
            raise ValueError("dt must be finite")
        if bool(torch.any(dt_tensor <= 0.0)):
            raise ValueError("dt must be positive")

        # Extract and align state variables
        theta = self._align_profile(working_state["soil_moisture"])
        porosity = self._align_profile(working_state["porosity"])
        temperature = self._align_profile(working_state["soil_temperature"])
        clay = self._align_profile(working_state["clay_content"])
        co2_conc = self._align_profile(working_state["soil_co2_concentration"])
        n2o_conc = self._align_profile(working_state["n2o_concentration"])
        
        # Calculate air-filled porosity
        clamp_min = float(
            self.get_parameter_value(
                "clamp_min", namespace="pedosphere.numerical", required=True
            )
        )
        air_porosity = torch.clamp(porosity - theta, min=clamp_min)

        dndc_factor = None
        if self.diffusivity_model == DiffusivityModel.DNDC:
            dndc_factor = self._dndc_n2o_diffusion_factor(
                theta / porosity,
                clay,
                porosity,
                temperature,
            )

        # Calculate effective diffusivity for each species
        co2_deff = self._calculate_effective_diffusivity(
            air_porosity, porosity, temperature, clay, "CO2"
        )
        n2o_deff = self._calculate_effective_diffusivity(
            air_porosity,
            porosity,
            temperature,
            clay,
            "N2O",
            dndc_factor=dndc_factor,
        )
        
        # Calculate diffusive fluxes
        co2_flux_profile, co2_surface_flux = self._calculate_diffusive_flux(
            concentration=co2_conc,
            deff=co2_deff,
            species="CO2"
        )
        
        n2o_flux_profile, n2o_surface_flux = self._calculate_diffusive_flux(
            concentration=n2o_conc,
            deff=n2o_deff,
            species="N2O"
        )
        
        # References: Li et al. (1992); Li (2016); Song et al. (2021); LeVeque (2002).
        # Mathematical expression: F_out,N2O = (C_surface - C_atm) dz D_N2O / t_h when
        #   pedosphere.transport.atmospheric_exchange is on; F_out,N2O = 0 with J_top = 0
        #   (closed top boundary) when the switch is off, exactly as in the Fickian branch.
        # Parameter meanings: C_surface is the top-layer N2O-N concentration, C_atm the
        #   atmospheric boundary concentration, dz the top-layer thickness, D_N2O the
        #   dimensionless DNDC factor, t_h the SSOT seconds per hour.
        # Units: C [g N m-3]; dz [m]; D_N2O [1]; t_h [s]; F_out [g N m-2 s-1].
        # Domain/constraints: the DNDC expression replaces only the Fickian surface term and
        #   must never reopen a disabled boundary; the interior flux profile is unchanged.
        # Sign convention: F_out is positive from soil to atmosphere.
        if (
            self.diffusivity_model == DiffusivityModel.DNDC
            and self.enable_atmospheric_exchange
        ):
            n2o_surface_flux = self._calculate_dndc_n2o_flux(
                n2o_conc,
                theta,
                clay,
                porosity,
                temperature,
                dndc_factor=dndc_factor,
            )
        
        # Ensure all outputs are tensors on correct device
        co2_surface_flux = self._ensure_tensor(co2_surface_flux)
        n2o_surface_flux = self._ensure_tensor(n2o_surface_flux)
        
        # Prepare canonical outputs in SCM-declared units.
        output = {
            "co2_surface_flux": co2_surface_flux,
            "n2o_surface_flux": n2o_surface_flux,
            "co2_flux_profile": co2_flux_profile,
            "n2o_flux_profile": n2o_flux_profile
        }
        
        # Standardize output keys
        output = standardize_output_keys(output)

        # Add concentration updates from flux divergence using unified operators
        # References: Fick's law J = -Deff dC/dz; Conservation dC/dt = -dJ/dz
        # Mathematical: delta_C_i = -divergence(J) * dt
        # Parameters: J [g C m-2 s-1 or g N m-2 s-1], dz [m], dt [s]
        dz_min = float(
            self.get_parameter_value(
                "epsilon_length", namespace="pedosphere.numerical", required=True
            )
        )
        dz = max(float(self.soil_depth) / float(self.n_layers), dz_min)

        # Helper function to calculate concentration changes from flux divergence
        def _calculate_concentration_change(
            flux_profile,
            surface_flux,
            current_concentration,
        ):
            """Return bounded concentration increment and signed clipping route."""
            if not isinstance(flux_profile, torch.Tensor) or not isinstance(surface_flux, torch.Tensor):
                return None, None

            # Construct complete flux array including surface and bottom
            # Surface flux at top, internal fluxes in middle, zero flux at bottom (no flow)
            # The public surface port is positive outward, while J is positive downward.
            # Therefore J_top = -F_surface,out and J_bottom = 0 for no bottom flow.
            J_top = -surface_flux.view(1)
            J_int = flux_profile.view(-1)
            J_bottom = torch.zeros_like(J_top, device=self.device)  # Zero flux at bottom
            J = torch.cat([J_top, J_int, J_bottom], dim=0)

            # References: LeVeque (2002), finite-volume conservation laws.
            # Mathematical expression: delta_C_i = -(J_i+1/2 - J_i-1/2) dt / dz.
            # Parameter meanings: J contains top, internal, and bottom boundary fluxes
            # [g element m-2 s-1], dz is layer thickness [m], and dt is [s].
            divergence = (J[1:] - J[:-1]) / dz
            raw_delta_c = -divergence * dt_tensor.to(
                device=divergence.device,
                dtype=divergence.dtype,
            )

            # References:
            #   - Fickian diffusion continuity: dC/dt = -dJ/dz
            # Mathematical expression:
            #   C_new = C_old + delta_C
            #   C_new >= 0 -> delta_C >= -C_old
            # Parameter meanings:
            #   current_concentration: layer concentration before update [g m-3]
            #   delta_c: concentration increment over one step [g m-3]
            if isinstance(current_concentration, torch.Tensor):
                c_old = current_concentration.view(-1)[:self.n_layers]
                zero_concentration = float(
                    self.get_parameter_value(
                        "zero_initial",
                        namespace="pedosphere.numerical",
                        required=True,
                    )
                )
                c_new = torch.clamp(c_old + raw_delta_c, min=zero_concentration)
                bounded_delta_c = c_new - c_old
                clipping_adjustment = bounded_delta_c - raw_delta_c
            else:
                bounded_delta_c = raw_delta_c
                clipping_adjustment = torch.zeros_like(raw_delta_c)

            return bounded_delta_c, clipping_adjustment

        # Calculate concentration changes for both species
        co2_delta_c, co2_clipping_adjustment = _calculate_concentration_change(
            output.get("co2_flux_profile"),
            output.get("co2_surface_flux"),
            co2_conc,
        )

        n2o_delta_c, n2o_clipping_adjustment = _calculate_concentration_change(
            output.get("n2o_flux_profile"),
            output.get("n2o_surface_flux"),
            n2o_conc,
        )

        # Add concentration updates to output if successful
        if co2_delta_c is not None:
            output["soil_co2_concentration"] = co2_delta_c
        if n2o_delta_c is not None:
            output["n2o_concentration"] = n2o_delta_c
        
        zero_delta = float(
            self.get_parameter_value(
                "zero_initial", namespace="pedosphere.numerical", required=True
            )
        )
        co2_delta_for_audit = output.get("soil_co2_concentration")
        if not isinstance(co2_delta_for_audit, torch.Tensor):
            co2_delta_for_audit = torch.full_like(co2_conc, zero_delta)
        n2o_delta_for_audit = output.get("n2o_concentration")
        if not isinstance(n2o_delta_for_audit, torch.Tensor):
            n2o_delta_for_audit = torch.full_like(n2o_conc, zero_delta)
        if isinstance(co2_delta_c, torch.Tensor) and isinstance(
            co2_clipping_adjustment, torch.Tensor
        ):
            co2_clipping_adjustment = co2_clipping_adjustment + (
                self._align_profile(co2_delta_for_audit) - co2_delta_c
            )
        else:
            co2_clipping_adjustment = torch.full_like(co2_conc, zero_delta)
        if isinstance(n2o_delta_c, torch.Tensor) and isinstance(
            n2o_clipping_adjustment, torch.Tensor
        ):
            n2o_clipping_adjustment = n2o_clipping_adjustment + (
                self._align_profile(n2o_delta_for_audit) - n2o_delta_c
            )
        else:
            n2o_clipping_adjustment = torch.full_like(n2o_conc, zero_delta)
        iphs_audit = self._iphs_audit_terms(
            co2_before=co2_conc,
            n2o_before=n2o_conc,
            co2_delta=self._align_profile(co2_delta_for_audit),
            n2o_delta=self._align_profile(n2o_delta_for_audit),
            co2_deff=co2_deff,
            n2o_deff=n2o_deff,
            co2_surface_flux=output["co2_surface_flux"],
            n2o_surface_flux=output["n2o_surface_flux"],
            dt=dt,
        )
        output["diagnostics.gas_diffusion.energy_storage"] = iphs_audit["energy_storage"]
        output["diagnostics.gas_diffusion.dissipation_rate"] = iphs_audit["dissipation_rate"]
        output["diagnostics.gas_diffusion.dH_dt_audit"] = iphs_audit["dH_dt_audit"]
        output["diagnostics.gas_diffusion.port_power_residual"] = iphs_audit[
            "port_power_residual"
        ]
        output[
            "diagnostics.gas_diffusion.co2_clipping_residual"
        ] = co2_clipping_adjustment
        output[
            "diagnostics.gas_diffusion.n2o_clipping_residual"
        ] = n2o_clipping_adjustment
        # Declared step length so the public boundary ledger can integrate the rates.
        output["diagnostics.gas_diffusion.step_seconds"] = dt_tensor.detach().to(
            dtype=n2o_clipping_adjustment.dtype
        ).reshape(())

        # Validate output
        validation_errors = validate_functor_output(
            output, self.get_output_specifications()
        )
        if validation_errors:
            raise ValueError(f"Output validation failed: {validation_errors}")

        units = {
            "co2_surface_flux": "g C m-2 s-1",
            "n2o_surface_flux": "g N m-2 s-1",
            "co2_flux_profile": "g C m-2 s-1",
            "n2o_flux_profile": "g N m-2 s-1",
            "soil_co2_concentration": "g C m-3",
            "n2o_concentration": "g N m-3",
            "diagnostics.gas_diffusion.energy_storage": "g2 m-6",
            "diagnostics.gas_diffusion.dissipation_rate": "g2 m-5 s-1",
            "diagnostics.gas_diffusion.dH_dt_audit": "g2 m-6 s-1",
            "diagnostics.gas_diffusion.port_power_residual": "g2 m-5 s-1",
            "diagnostics.gas_diffusion.co2_clipping_residual": "g C m-3",
            "diagnostics.gas_diffusion.n2o_clipping_residual": "g N m-3",
            "diagnostics.gas_diffusion.step_seconds": "s",
        }
        state_increments = {"soil_co2_concentration", "n2o_concentration"}
        elements = {
            "co2_surface_flux": "C",
            "co2_flux_profile": "C",
            "soil_co2_concentration": "C",
            "n2o_surface_flux": "N",
            "n2o_flux_profile": "N",
            "n2o_concentration": "N",
        }
        return {
            key: UpdateItem(
                value=value,
                mode=(
                    UpdateMode.ACCUMULATE
                    if key in state_increments
                    else UpdateMode.REPLACE
                ),
                unit=units[key],
                basis="per_step" if key in state_increments else "instantaneous",
                per_timestep=key in state_increments,
                element=elements.get(key),
                source_functor=self.name,
            )
            for key, value in output.items()
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
        dt: Optional[torch.Tensor | float] = None,
    ) -> Dict[str, float]:
        """Return the maximum C/N finite-volume boundary-ledger residual.

        References: Fick (1855); LeVeque (2002), finite-volume conservation.
        Mathematical expression: sum(delta C) dz + F_out dt - sum(R_clip) dz = 0.
        Parameter meanings: delta C is the accepted layer increment [g element
        m-3], F_out is the outward-positive surface flux [g element m-2 s-1],
        R_clip is the signed non-negativity adjustment [g element m-3], dz is
        layer thickness [m], and dt is the time step [s].
        """
        routes = (
            (
                "soil_co2_concentration",
                "co2_surface_flux",
                "diagnostics.gas_diffusion.co2_clipping_residual",
            ),
            (
                "n2o_concentration",
                "n2o_surface_flux",
                "diagnostics.gas_diffusion.n2o_clipping_residual",
            ),
        )
        required_outputs = {key for route in routes for key in route}
        if not required_outputs.issubset(output_updates):
            return {}

        def _payload(key: str) -> torch.Tensor:
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            tensor = self._ensure_tensor(value)
            if not bool(torch.all(torch.isfinite(tensor))):
                raise ValueError(f"{key} must be finite for conservation audit")
            return tensor

        if dt is not None:
            dt_value = self._ensure_tensor(dt)
        else:
            step_key = "diagnostics.gas_diffusion.step_seconds"
            if step_key not in output_updates:
                raise ValueError("dt is required for the gas boundary ledger")
            dt_value = _payload(step_key)
        if dt_value.numel() != 1 or not bool(torch.all(torch.isfinite(dt_value))):
            raise ValueError("dt is required for the gas boundary ledger")
        if float(dt_value.detach().reshape(-1)[0].item()) <= 0.0:
            raise ValueError("dt is required for the gas boundary ledger")
        dz = self.layer_thickness[0]

        residuals = []
        for concentration_key, surface_flux_key, clipping_key in routes:
            concentration_delta = _payload(concentration_key)
            surface_flux = _payload(surface_flux_key)
            clipping_adjustment = _payload(clipping_key)
            dt_tensor = dt_value.to(
                device=concentration_delta.device,
                dtype=concentration_delta.dtype,
            )
            dz_tensor = dz.to(
                device=concentration_delta.device,
                dtype=concentration_delta.dtype,
            )
            residuals.append(
                torch.sum(concentration_delta) * dz_tensor
                + surface_flux.reshape(-1)[0] * dt_tensor
                - torch.sum(clipping_adjustment) * dz_tensor
            )

        magnitude = max(
            float(torch.max(torch.abs(residual)).detach().item())
            for residual in residuals
        )
        return {law: magnitude for law in self.metadata.conservation_laws}

    def _dndc_n2o_diffusion_factor(
        self,
        wfps: torch.Tensor,
        clay: torch.Tensor,
        porosity: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """Return the dimensionless DNDC N2O diffusion factor.

        References:
          - Li et al. (1992), doi:10.1029/92JD00509.
          - Song et al. (2021), GMD discussion supplement, Table S1, Eq. 20.
        Mathematical expression:
          D_N2O = (1 - WFPS) * [a0 + ac*Cclay
              + (ap - acp*Cclay) * b^(Tc/Tscale) * phi].
        Parameter meanings:
          WFPS, Cclay, phi, a0, ac, ap, acp, and b are dimensionless;
          Tc and Tscale are temperature intervals in K-equivalent units.
        """

        def _coefficient(name: str) -> torch.Tensor:
            value = self.get_parameter_value(
                name,
                namespace="pedosphere.transport",
                required=True,
            )
            return self._ensure_tensor(value).to(
                device=wfps.device,
                dtype=wfps.dtype,
            )

        wfps_min = float(
            self.get_parameter_value(
                "wfps_minimum", namespace="pedosphere.numerical", required=True
            )
        )
        wfps_max = float(
            self.get_parameter_value(
                "wfps_maximum", namespace="pedosphere.numerical", required=True
            )
        )
        bounded_wfps = torch.clamp(wfps, wfps_min, wfps_max)
        temp_c = temperature - self.celsius_offset
        base_coefficient = _coefficient("n2o_dndc_base_coefficient")
        clay_coefficient = _coefficient("n2o_dndc_clay_effect")
        porosity_coefficient = _coefficient("n2o_dndc_porosity_effect")
        clay_porosity_coefficient = _coefficient(
            "n2o_dndc_clay_porosity_coefficient"
        )
        temperature_base = _coefficient("n2o_dndc_temperature_base")
        temperature_divisor = _coefficient("n2o_dndc_temperature_divisor")
        temperature_factor = torch.pow(
            temperature_base,
            temp_c / temperature_divisor,
        )
        return (1.0 - bounded_wfps) * (
            base_coefficient
            + clay_coefficient * clay
            + (
                porosity_coefficient
                - clay_porosity_coefficient * clay
            )
            * temperature_factor
            * porosity
        )
    
    def _calculate_effective_diffusivity(self,
                                        air_porosity: torch.Tensor,
                                        total_porosity: torch.Tensor,
                                        temperature: torch.Tensor,
                                        clay: torch.Tensor,
                                        species_name: str,
                                        dndc_factor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate effective gas diffusivity in soil.
        
        Parameters:
        -----------
        air_porosity : torch.Tensor
            Air-filled porosity [m^3 m^-3]
        total_porosity : torch.Tensor
            Total porosity [m^3 m^-3]
        temperature : torch.Tensor
            Temperature [K]
        clay : torch.Tensor
            Clay fraction [-]
        species_name : str
            Gas species ("CO2" or "N2O")
            
        Returns:
        --------
        torch.Tensor
            Effective diffusivity [m^2 s^-1]
        """
        species = self.species[species_name]
        
        # Convert species parameters to tensors
        d0_at_stp = self._ensure_tensor(species.d0_at_stp)
        temp_exp = self._ensure_tensor(species.temperature_exponent)
        
        # Base diffusivity in free air (temperature corrected)
        # References: Fuller et al. (1966), temperature dependence of gas diffusion
        # Mathematical expressions: D(T) = D_ref * (T/T_ref)^n
        # Parameters: reference temperature [K], temperature exponent
        if self.enable_temperature_correction:
            temp_factor = torch.pow(
                temperature / self.reference_temperature,
                temp_exp,
            )
            d0 = d0_at_stp * temp_factor
        else:
            # Ensure d0 has same shape as temperature but constant value
            if d0_at_stp.numel() == 1:
                d0 = d0_at_stp.expand_as(temperature)
            else:
                d0 = d0_at_stp
        
        # Effective diffusivity based on selected model
        if self.diffusivity_model == DiffusivityModel.MILLINGTON_QUIRK:
            # Millington-Quirk model: Deff = D0 * thetaa^(10/3) / phi^2
            # References: Millington & Quirk (1961), empirical tortuosity model
            # Mathematical expressions: tortuosity = (theta_a/phi)^(10/3) / phi
            # Parameters: numerator exponent (10/3), denominator exponent (2)
            num_exp = self.get_parameter_value(
                "millington_quirk_exponent_numerator",
                namespace="pedosphere.physical", required=True)
            denom_exp = self.get_parameter_value(
                "millington_quirk_exponent_denominator",
                namespace="pedosphere.physical", required=True)
            tortuosity = torch.pow(air_porosity, num_exp) / torch.pow(total_porosity, denom_exp)
            deff = d0 * tortuosity
            
        elif self.diffusivity_model == DiffusivityModel.PENMAN:
            # Penman model: Deff = D0 * coeff * thetaa
            # References: Penman (1940), empirical gas diffusion coefficient
            # Mathematical expressions: D_eff = D0 * coefficient * air_porosity
            # Parameters: Penman coefficient [dimensionless]
            penman_coeff = self.get_parameter_value(
                "penman_coefficient",
                namespace="pedosphere.physical", required=True)
            deff = d0 * penman_coeff * air_porosity
            
        elif self.diffusivity_model == DiffusivityModel.DNDC:
            # DNDC model (specific for N2O)
            if species_name == "N2O":
                factor = dndc_factor
                if factor is None:
                    factor = self._dndc_n2o_diffusion_factor(
                        1.0 - air_porosity / total_porosity,
                        clay,
                        total_porosity,
                        temperature,
                    )

                # Map the dimensionless DNDC factor to the hourly layer operator.
                # References: Song et al. (2021), Table S1, Eq. 20.
                # Mathematical expression: D_eff = D_N2O * dz^2 / seconds_per_hour.
                # Parameters: D_N2O [-], dz [m], seconds_per_hour [s h-1].
                layer_depth = float(self.soil_depth) / float(self.n_layers)
                deff = factor * layer_depth**2 / self.seconds_per_hour
            else:
                # default to Millington-Quirk for other gases
                num_exp = self.get_parameter_value(
                    "millington_quirk_exponent_numerator",
                    namespace="pedosphere.physical", required=True)
                denom_exp = self.get_parameter_value(
                    "millington_quirk_exponent_denominator",
                    namespace="pedosphere.physical", required=True)
                tortuosity = torch.pow(air_porosity, num_exp) / torch.pow(total_porosity, denom_exp)
                deff = d0 * tortuosity
        else:
            raise ValueError(f"Unknown diffusivity model: {self.diffusivity_model}")
        
        # Ensure minimum diffusivity
        clamp_min = float(
            self.get_parameter_value(
                "clamp_min", namespace="pedosphere.numerical", required=True
            )
        )
        deff = torch.clamp(deff, min=clamp_min)
        
        return deff
    
    def _calculate_diffusive_flux(self,
                                 concentration: torch.Tensor,
                                 deff: torch.Tensor,
                                 species: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate diffusive gas flux using unified transport operators.

        Fully refactored to use unified operators from operators.py for all calculations.
        Implements Fick's law: J = -Deff partialC/partialz

        References:
        -----------
        - Fick's first law of diffusion: J = -D * gradient(C)
        - Harmonic mean for interface properties: D_interface = 2*D1*D2/(D1+D2)

        Parameters:
        -----------
        concentration : torch.Tensor
            Gas concentration profile [g m^-3]
        deff : torch.Tensor
            Effective diffusivity profile [m^2 s^-1]
        species : str
            Gas species name

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            (flux_profile, surface_flux)
            flux_profile: Fluxes between layers [g m^-2 s^-1]
            surface_flux: Surface emission flux [g m^-2 s^-1]
        """
        # Layer thickness (assuming uniform layers)
        dz = self.layer_thickness[0]

        # Use unified operators for gradient calculation
        # This computes dC/dz at each layer center using central differences
        # Mathematical expression: gradient = (C[i+1] - C[i-1]) / (2*dz)
        gradient_operator = OperatorFactory.create(
            "gradient.central_difference_1d",
            name="gradient_calc",
            config={
                "dx": float(dz),
                "boundary_method": "extrapolate",
                "order": 2
            },
            validate_parameters=False
        )
        gradient = gradient_operator.apply(
            field=concentration.unsqueeze(0),  # Add batch dimension for operators API
            dx=float(dz),
            boundary_method="extrapolate",
            order=2
        ).squeeze(0)  # Remove batch dimension

        # Calculate flux at layer interfaces using unified harmonic mean operator
        # Replace the explicit loop with vectorized operations where possible
        flux_profile = torch.zeros(self.n_layers - 1, device=self.device)

        # Vectorized interface flux calculation using harmonic_mean from operators.py
        deff_pairs = torch.stack([deff[:-1], deff[1:]], dim=1)  # Shape: (n_layers-1, 2)
        deff_interfaces = harmonic_mean(deff_pairs[:, 0], deff_pairs[:, 1])

        # Vectorized gradient averaging at interfaces
        grad_interfaces = (gradient[:-1] + gradient[1:]) / 2

        # Apply Fick's law: J = -Deff * gradient
        # Mathematical: flux_interface = -D_interface * avg_gradient
        flux_profile = -deff_interfaces * grad_interfaces

        # Surface flux calculation with unified operators
        if self.enable_atmospheric_exchange:
            species_config = self.species[species]
            atmospheric_value = species_config.atmospheric_concentration
            if atmospheric_value is None:
                raise ValueError(
                    f"Missing atmospheric concentration boundary for {species}"
                )
            atm_conc = self._ensure_tensor(
                atmospheric_value
            )

            # Create extended concentration array with atmospheric value
            # This provides proper boundary condition for surface flux calculation
            conc_with_atm = torch.cat([atm_conc.unsqueeze(0), concentration])

            # Calculate surface gradient using unified gradient operator
            # Mathematical: gradient_surface = (C[1] - C_atm) / dz
            gradient_operator = OperatorFactory.create(
                "gradient.central_difference_1d",
                name="surface_gradient_calc",
                config={
                    "dx": float(dz/2),
                    "boundary_method": "extrapolate",
                    "order": 2
                },
                validate_parameters=False
            )
            surface_gradient = gradient_operator.apply(
                field=conc_with_atm.unsqueeze(0),
                dx=float(dz/2),
                boundary_method="extrapolate",
                order=2
            ).squeeze(0)[0]  # Get gradient at first point (soil surface)

            # References:
            #   - Millington and Quirk (1961), doi:10.1039/TF9615701200
            #   - Kusa et al. (2008), doi:10.1111/j.1747-0765.2008.00292.x
            # Mathematical expression:
            #   J_z = -D_eff_surface * dC/dz; F_out = -J_z.
            # Parameter meanings:
            #   deff[0]: effective gas diffusivity at top soil layer [m2 s-1]
            #   surface_gradient: concentration gradient at soil-atmosphere interface [g m-4]
            surface_flux = deff[0] * surface_gradient
        else:
            # No atmospheric exchange: zero surface flux
            surface_flux = torch.zeros_like(concentration[0])

        return flux_profile, surface_flux
    
    def _calculate_dndc_n2o_flux(self,
                                n2o_conc: torch.Tensor,
                                theta: torch.Tensor,
                                clay: torch.Tensor,
                                porosity: torch.Tensor,
                                temperature: torch.Tensor,
                                dndc_factor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate N2O flux using DNDC formulation.
        
        Special formulation for N2O that considers WFPS, clay content,
        and temperature effects.

        References:
        -----------
        Li et al. (1992), doi:10.1029/92JD00509.
        Li et al. (2000) and Li (2016), DNDC N2O diffusion coefficient.
        Song et al. (2021), GMD discussion supplement, Table S1, Eq. 20.

        Mathematical expression:
        ------------------------
        D_N2O = (1 - WFPS) * [a0 + ac*Cclay
            + (ap - acp*Cclay) * b^(Tc/Tscale) * phi]
        F_out = (Csurface - Catm) * dz * D_N2O / seconds_per_hour

        Parameter meanings:
        -------------------
        D_N2O is the dimensionless empirical diffusion factor.
        Csurface and Catm are elemental N concentrations [g N m^-3].
        dz is the surface-layer thickness [m].
        F_out is positive from soil to atmosphere [g N m^-2 s^-1].
        
        Parameters:
        -----------
        n2o_conc : torch.Tensor
            N2O-N concentration profile [g N m^-3]
        theta : torch.Tensor
            Soil water content [m^3 m^-3]
        clay : torch.Tensor
            Clay fraction [-]
        porosity : torch.Tensor
            Total porosity [m^3 m^-3]
        temperature : torch.Tensor
            Temperature [K]
            
        Returns:
        --------
        torch.Tensor
            N2O surface flux [g N m^-2 s^-1]
        """
        factor = dndc_factor
        if factor is None:
            factor = self._dndc_n2o_diffusion_factor(
                theta / porosity,
                clay,
                porosity,
                temperature,
            )

        atmospheric_concentration = self.species["N2O"].atmospheric_concentration
        if atmospheric_concentration is None:
            raise ValueError("Atmospheric N2O concentration is required for surface exchange.")
        atmospheric_concentration_tensor = self._ensure_tensor(
            atmospheric_concentration
        ).to(device=n2o_conc.device, dtype=n2o_conc.dtype)
        layer_depth = float(self.soil_depth) / float(self.n_layers)
        concentration_excess = n2o_conc[0] - atmospheric_concentration_tensor
        return (
            concentration_excess
            * layer_depth
            * factor[0]
            / self.seconds_per_hour
        )
    
# Export
__all__ = ['GasDiffusion1D', 'GasSpecies', 'DiffusivityModel']
