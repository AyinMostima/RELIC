"""
Soil-Vegetation Coupling Module

This module implements the scale coupling from a resolved soil-water profile to
root water uptake. It evaluates the Feddes reduction law and exposes the uptake
profile as an explicit port for a separately selected Richards or bucket water
closure. It does not advance soil moisture itself.

The coupling process involves:
1. Root water uptake as a function of soil water potential and root distribution
2. Normalization of the uptake profile to the actual transpiration boundary flux
3. Feedback between transpiration demand and available soil water
4. Root-zone observations for explicit downstream coupling

Mathematical Framework:
- Feddes model: S(z) = alpha(psi) * beta(z) * T_p
- Root distribution: beta(z) = a * exp(-a * z) (exponential decay)
- Water potential stress: alpha(psi) = piecewise function of matric potential
- Port balance: integral S(z) dz = T_actual <= T_potential

Units and Variables (CF-1.8 compliant):
- S(z): root water uptake sink [kg m-3 s-1]
- psi: soil matric potential [Pa]
- theta: soil moisture content [m3 m-3]
- T_p: potential transpiration [kg m-2 s-1]
- beta(z): root distribution function [m-1]

Conservation Laws:
- Water mass conservation: integral S(z) dz = T_actual <= T_potential
- Root uptake constraint: S(z) >= 0 (no water exudation)
- This coupling exposes a sink port and never advances soil moisture locally

References:
- Feddes, R.A., Kowalik, P.J., Zaradny, H. (1978). Simulation of field water use 
  and crop yield. Pudoc, Wageningen
- Skaggs, T.H. et al. (2006). Macroscopic approaches to root water uptake as a
  function of water and salinity stress. Agricultural Water Management 86: 140-149
- Jackson, R.B. et al. (1996). A global analysis of root distributions for 
  terrestrial biomes. Oecologia 108: 389-411
"""

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor

from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.functor import ScaleCouplingFunctor
from relic.core.abstractions import StateDict, FluxDict, DiagnosticsDict
from relic.core.state import UpdateItem, UpdateMode
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace
from relic.coupling import SV_NAMESPACE
from relic.coupling.core.conservation_checker import require_relative_tolerance
from relic.coupling.scales.conservation_enforcer import ConservationEnforcer
from relic.coupling.internal.common import (
    CouplingParameterMixin,
)
from relic.coupling.internal.dependency_utils import (
    extract_dependency_overrides,
    validate_dependency_names,
)

# Scientific metadata constants
MATHEMATICAL_EXPRESSION_SV = (
    "Feddes: S(z) = alpha(psi) * beta(z) * T_p; "
    "Root distribution: beta(z) = a * exp(-a * z); "
    "Water stress: alpha(psi) = piecewise function of matric potential; "
    "Port balance: integral S(z) dz = T_actual <= T_potential"
)

REFERENCES_SV = [
    "Feddes, R.A., Kowalik, P.J., Zaradny, H. (1978). Simulation of field water use and crop yield",
    "Skaggs, T.H. et al. (2006). Macroscopic approaches to root water uptake as a function of water and salinity stress",
    "Jackson, R.B. et al. (1996). A global analysis of root distributions for terrestrial biomes",
]

# Resolved namespaces (module-local to avoid repeated resolution and enum/string mixing)
CONSERVATIVE_NS = ParameterNamespace.COUPLING_SCALES_CONSERVATIVE.value
VALIDATION_NUM_NS = ParameterNamespace.VALIDATION_NUMERICAL.value
VALIDATION_CONSERVATION_NS = ParameterNamespace.COUPLING_CONSERVATION.value


@dataclass
class RootDistributionParams:
    """
    Parameters for root distribution models.
    
    Attributes:
        distribution_type : str
            Root distribution model ("exponential", "linear", "beta")
        root_depth_max : float
            Maximum rooting depth [m]
        shape_parameter : float
            Shape parameter for distribution (varies by type)
    """
    distribution_type: str
    root_depth_max: float
    shape_parameter: float


@dataclass
class SoilWaterStressParams:
    """
    Parameters for water stress response functions.
    
    Attributes:
        wilting_point : float
            Permanent wilting point matric potential [Pa] or [m]
        anaerobic_point : float
            Anaerobic stress threshold [Pa] or [m]
        optimal_range : Tuple[float, float]
            Optimal water potential range [Pa] or [m]
    """
    wilting_point: float
    anaerobic_point: float
    optimal_range: Tuple[float, float]


class SoilVegetationCoupling(CouplingParameterMixin, ScaleCouplingFunctor):
    """
    Soil-Vegetation Coupling Functor
    
    Implements the Feddes root water uptake model as an explicit profile-to-port
    coupling between soil water availability and plant transpiration demand.
    
    Key Features:
    - Four-threshold Feddes root water uptake reduction function
    - Flexible root distribution profiles (exponential, linear, beta)
    - Explicit uptake port for a separately selected soil-water closure
    - Namespaced Feddes stress and root-zone observations
    - Layer-resolved water balance between uptake and transpiration
    
    Input Variables:
    - soil_moisture_profile: Soil moisture by layer [m3 m-3]
    - soil_matric_potential: Soil water potential by layer [Pa]
    - potential_transpiration: Atmospheric transpiration demand [kg m-2 s-1]
    - soil_depth_layers: Soil layer-bottom depths z_i, top at 0 [m]
    - root_density_profile: Root density by layer [kg m-3] (optional)
    
    Output Variables:
    - root_uptake_profile: Water uptake by soil layer [kg m-3 s-1]
    - effective_transpiration: Actual transpiration achieved [kg m-2 s-1]
    - diagnostics.soil_vegetation.feddes_stress_factor: Root-weighted stress [1]
    - diagnostics.soil_vegetation.root_zone_moisture: Root-zone moisture [m3 m-3]
    - diagnostics.soil_vegetation.root_distribution: Normalized profile [m-1]
    - diagnostics.soil_vegetation.water_balance_residual: Port residual [kg m-2 s-1]
    
    Mathematical Expressions:
    Feddes Root Uptake Model:
        S(z) = alpha(psi(z)) * beta(z) * T_p
        
        where:
        - alpha(psi): water stress reduction factor [1]
        - beta(z): normalized root distribution [m-1]
        - T_p: potential transpiration [kg m-2 s-1]
    
    Root Distribution Functions:
        Exponential: beta(z) = a * exp(-a * z) where a is shape parameter
        Linear: beta(z) = 2(1 - z/D)/D where D is maximum depth
        Beta: beta(z) = (z/D)^(p-1) * (1-z/D)^(q-1) / B(p,q) normalized
    
    Water Stress Function (Feddes):
        alpha(psi) = {
            0                               if psi < psi_wilt or psi > psi_anaer
            (psi - psi_wilt)/(psi_opt1 - psi_wilt) if psi_wilt <= psi < psi_opt1
            1                               if psi_opt1 <= psi <= psi_opt2
            (psi_opt2 - psi)/(psi_opt2 - psi_anaer) if psi_opt2 < psi <= psi_anaer
        }
    
    Water Port Balance:
        integral S(z) dz - T_actual = 0
        where the selected soil-water closure consumes S(z) as its sink port.

    Parameters
    ----------
    name : str
        Functor name.
    config : Dict
        Configuration for soil-vegetation coupling.
    metadata : Optional[FunctorMetadata]
        Scientific metadata (mathematical_expression, references, units).
    parameter_manager : Optional[Any]
        Parameter manager instance for parameter retrieval (units validated).

    Returns
    -------
    Dict[str, UpdateItem]
        Coupled soil-plant flux ports and namespaced observations.

    References
    ----------
    Feddes, R.A. et al. (1978) root water uptake model.
    Skaggs, T.H. et al. (2006) macroscopic root water uptake.
    Jackson, R.B. et al. (1996) global root distributions.
    """
    
    SOIL_NAMESPACE = SV_NAMESPACE
    DEPENDENCY_OVERRIDE_KEYS = ("depends_on",)

    @staticmethod
    def _to_scalar(value: Any) -> float:
        return ScaleCouplingFunctor._to_scalar(value)

    def __init__(self, name: str, config: Dict, metadata: Optional[FunctorMetadata] = None,
                 parameter_manager: Optional[Any] = None):
        """
        Initialize soil-vegetation coupling functor.
        
        Parameters:
            name : str
                Functor name
            config : Dict
                Configuration containing root and stress parameters
            metadata : Optional[FunctorMetadata]
                Scientific metadata
        """
        if metadata is None:
            metadata = FunctorMetadata(
                name=name,
                version="1.0.0",
                description="Maps soil-water profiles and transpiration demand to a Feddes uptake port",
                category=ProcessCategory.SCALE_COUPLING,
                execution_phase=ExecutionPhase.COUPLING,
                mathematical_expression=MATHEMATICAL_EXPRESSION_SV,
                references=REFERENCES_SV,
                parameters={
                    "root_distribution_type": "Root distribution model (exponential, linear, beta)",
                    "root_depth_max": "Maximum rooting depth [m]",
                    "root_shape_parameter": "Shape parameter for root distribution [m-1]",
                    "wilting_point": "Permanent wilting point matric potential [Pa]",
                    "optimal_water_potential": "Optimal water potential range [Pa]",
                    "anaerobic_point": "Wet-end anaerobiosis threshold [Pa]",
                },
                input_variables=[
                    "soil_moisture_profile", "soil_matric_potential", "potential_transpiration",
                    "soil_depth_layers", "root_density_profile"
                ],
                output_variables=[
                    "root_uptake_profile",
                    "effective_transpiration",
                    "diagnostics.soil_vegetation.feddes_stress_factor",
                    "diagnostics.soil_vegetation.root_zone_moisture",
                    "diagnostics.soil_vegetation.root_distribution",
                    "diagnostics.soil_vegetation.water_balance_residual",
                ],
                # The profile-to-canopy water ledger is checked explicitly because
                # the generic local checker cannot integrate a volumetric sink by
                # layer thickness before comparing it with an areal flux.
                conservation_laws=[],
                spatial_dimensions=1,
                temporal_scale=TemporalScale.SECOND,
                coupling_interfaces=[
                    "soil_water_profile",
                    "transpiration_demand",
                    "root_uptake_port",
                ],
                spatial_scale=SpatialScale.GRID_CELL,
                variable_kind=VariableKind.FLUX,
            )
        
        # The Feddes uptake maps the soil-layer profile (theta_i, psi_i, root density by layer)
        # to the grid-cell column (transpiration demand and effective transpiration): the
        # vertical soil_layer -> grid_cell spatial coupling.
        super().__init__(
            name,
            config,
            metadata,
            parameter_manager=parameter_manager,
            source_scale="soil_layer",
            target_scale="grid_cell",
            coupling_type="spatial",
        )
        self.optional_input_variables = {"root_density_profile"}
        # The canonical ``depends_on`` override, when given, replaces the
        # config-declared predecessors; the result is the instance declaration,
        # merged with later add_dependency calls by the bridge default.
        configured_dependencies = extract_dependency_overrides(
            config, self.DEPENDENCY_OVERRIDE_KEYS
        )
        declared_dependencies = validate_dependency_names(self._dependencies or [])
        self.set_dependencies(
            configured_dependencies
            if configured_dependencies
            else declared_dependencies
        )
        clamp_namespace = CONSERVATIVE_NS
        self.zero_clamp = self._to_scalar(
            self.get_parameter_value(
                "clamp_min",
                namespace=clamp_namespace,
                units="dimensionless",
                validate=True,
            )
        )
        self.unit_clamp = self._to_scalar(
            self.get_parameter_value(
                "clamp_max",
                namespace=clamp_namespace,
                units="dimensionless",
                validate=True,
            )
        )
        # The water ledger of this coupling is a domain budget check: its one key is
        # validation.conservation.conservation_tolerance (validation/conservation.yaml).
        self.conservation_tolerance = require_relative_tolerance(
            self._to_scalar(
                self.get_parameter_value(
                    "conservation_tolerance",
                    namespace=VALIDATION_CONSERVATION_NS,
                    units="dimensionless",
                    validate=True,
                    required=True,
                )
            ),
            f"{VALIDATION_CONSERVATION_NS}.conservation_tolerance",
        )
        self.positive_epsilon = self._to_scalar(
            self.get_parameter_value(
                "epsilon_divide",
                namespace=VALIDATION_NUM_NS,
                units="dimensionless",
                validate=True,
            )
        )
        self._conservation_enforcer = ConservationEnforcer(
            conservation_properties=["water"],
            device=self.device,
            track_history=False,
        )
        
        soil_ns = resolve_namespace(self.SOIL_NAMESPACE)
        # Load root distribution parameters from ParameterManager
        self.root_params = RootDistributionParams(
            distribution_type=str(
                self._get_coupling_parameter(
                    "distribution_type",
                    namespace=soil_ns,
                    units="dimensionless",
                    required=True,
                    as_scalar=False,
                )
            ),
            root_depth_max=float(
                self.get_parameter_value(
                    "root_depth_max",
                    namespace=soil_ns,
                    units="m",
                    validate=True,
                    required=True,
                )
            ),
            shape_parameter=float(
                self.get_parameter_value(
                    "shape_parameter",
                    namespace=soil_ns,
                    units="m-1",
                    validate=True,
                    required=True,
                )
            ),
        )

        optimal_range_raw = self._get_coupling_parameter(
            "optimal_range", namespace=soil_ns, units="Pa", expected_type=list
        )
        optimal_range = tuple(float(v) for v in optimal_range_raw) if optimal_range_raw else tuple()
        if len(optimal_range) != 2:
            raise ValueError(
                "coupling.scales.soil_vegetation.optimal_range must provide two bounds (Pa)"
            )
        self.stress_params = SoilWaterStressParams(
            wilting_point=float(
                self._get_coupling_parameter("wilting_point_potential", namespace=soil_ns, units="Pa")
            ),
            anaerobic_point=float(
                self._get_coupling_parameter("anaerobic_point", namespace=soil_ns, units="Pa")
            ),
            optimal_range=optimal_range,
        )
        optimum_dry, optimum_wet = self.stress_params.optimal_range
        if not (
            self.stress_params.wilting_point
            < optimum_dry
            < optimum_wet
            < self.stress_params.anaerobic_point
        ):
            raise ValueError(
                "Feddes thresholds must satisfy wilting < optimum_dry "
                "< optimum_wet < anaerobic in Pa"
            )
        self.linear_distribution_coeff = float(
            self._get_coupling_parameter(
                "linear_distribution_coeff",
                namespace=soil_ns,
                units="dimensionless",
            )
        )
        self.beta_distribution_q = float(
            self._get_coupling_parameter(
                "beta_q_parameter",
                namespace=soil_ns,
                units="dimensionless",
            )
        )

    def compute_root_distribution(
        self,
        depth_profile: Optional[Tensor] = None,
        root_density: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute normalized root distribution beta(z).
        
        Parameters:
            depth_profile : Tensor
                Layer bottom depths measured from surface [m]; top boundary implicitly 0
            root_density : Optional[Tensor]
                Observed root density profile [kg m-3] (overrides model)
                
        Returns:
            Tensor: Normalized root distribution [m-1]
            
        Mathematical Expressions:
            Exponential: beta(z) = a * exp(-a * z)
            Linear: beta(z) = 2(1 - z/D) / D
            Beta: beta(z) = normalized beta distribution
        """
        if depth_profile is None:
            raise ValueError("depth_profile is required")
        depth_profile = torch.as_tensor(depth_profile, dtype=torch.float32)
        if depth_profile.ndim != 1 or depth_profile.numel() < 1:
            raise ValueError("depth_profile must be a non-empty one-dimensional tensor")
        if not bool(torch.all(torch.isfinite(depth_profile))):
            raise ValueError("depth_profile must contain only finite values")
        if bool(torch.any(depth_profile <= 0.0)) or (
            depth_profile.numel() > 1
            and bool(torch.any(torch.diff(depth_profile) <= 0.0))
        ):
            raise ValueError("depth_profile must contain strictly increasing layer bottoms")
        if root_density is not None:
            root_density = torch.as_tensor(
                root_density, device=depth_profile.device, dtype=depth_profile.dtype
            )
            if root_density.shape != depth_profile.shape:
                raise ValueError("root_density must match the depth_profile shape")
            if not bool(torch.all(torch.isfinite(root_density))):
                raise ValueError("root_density must contain only finite values")
            if bool(torch.any(root_density < 0.0)):
                raise ValueError("root_density must be non-negative")

        # Enforce layer-bottom semantics: values are cumulative depths to layer bottoms
        boundaries = torch.cat(
            [
                depth_profile.new_zeros(1),
                depth_profile,
            ]
        )
        layer_thickness = torch.diff(boundaries)  # n layer thicknesses (m)

        if root_density is not None:
            # References: Jackson et al. (1996) global root profiles
            # Mathematical expression: beta(z) = rho_root(z) / integralrho_root(z) dz
            total_root_mass = torch.sum(root_density * layer_thickness)
            if not bool(total_root_mass > self.positive_epsilon):
                raise ValueError("root_density must have positive depth-integrated mass")
            return root_density / total_root_mass

        # Model-based root distribution using layer centers derived from boundaries
        # References: Feddes et al. (1978) S(z) = alpha(psi) * beta(z) * T_p
        z = (boundaries[:-1] + boundaries[1:]) / 2.0  # layer centers [m]
        
        max_depth = self.root_params.root_depth_max
        
        # Ensure depths don't exceed maximum rooting depth
        z_norm = torch.clamp(z / max_depth, min=self.zero_clamp, max=self.unit_clamp)
        
        if self.root_params.distribution_type == "exponential":
            # Exponential decay: beta(z) = a * exp(-a * z)
            a = self.root_params.shape_parameter
            beta_z = a * torch.exp(-a * z_norm * max_depth)

        elif self.root_params.distribution_type == "linear":
            # Linear decrease: beta(z) = coeff(1 - z/D) / D  
            beta_z = self.linear_distribution_coeff * (1.0 - z_norm) / max_depth
            beta_z = torch.clamp(beta_z, min=self.zero_clamp)

        elif self.root_params.distribution_type == "beta":
            # Beta distribution (flexible shape)
            p = self.root_params.shape_parameter
            q = self.beta_distribution_q
            beta_z = torch.pow(z_norm, p-1) * torch.pow(1.0 - z_norm, q-1)
            
        else:
            raise ValueError(f"Unknown root distribution type: {self.root_params.distribution_type}")
        
        # Normalize to ensure integration over depth equals 1
        # boundaries include surface at 0; layer_thickness already computed from boundaries
        total_integral = torch.sum(beta_z * layer_thickness)
        if not bool(total_integral > self.positive_epsilon):
            raise ValueError("Root distribution has zero depth-integrated support")
        beta_z_normalized = beta_z / total_integral
        
        return beta_z_normalized
    
    def compute_water_stress_factor(self, matric_potential: Tensor) -> Tensor:
        """
        Compute water stress reduction factor alpha(psi) using Feddes model.

        References:
            Feddes et al. (1978); Skaggs et al. (2006), Eq. 13; HYDRUS
            Technical Manual, root-water-uptake stress response.
        
        Parameters:
            matric_potential : Tensor
                Soil matric potential [Pa] (negative values)
                
        Returns:
            Tensor: Water stress factor [1] (0 = full stress, 1 = no stress)
            
        Mathematical Expression and Parameter Meanings:
            With psi_wilt < psi_dry < psi_wet < psi_anaerobic [Pa],
            alpha(psi) = {
                0, if psi <= psi_wilt or psi >= psi_anaerobic
                (psi-psi_wilt)/(psi_dry-psi_wilt), if psi_wilt < psi < psi_dry
                1, if psi_dry <= psi <= psi_wet
                (psi_anaerobic-psi)/(psi_anaerobic-psi_wet),
                    if psi_wet < psi < psi_anaerobic
            }
            psi is soil matric potential, psi_wilt is the dry no-uptake
            threshold, [psi_dry, psi_wet] is the optimal interval, and
            psi_anaerobic is the wet no-uptake threshold. alpha is dimensionless.
        """
        psi = torch.as_tensor(matric_potential)
        if not bool(torch.all(torch.isfinite(psi))):
            raise ValueError("matric_potential must contain only finite values")

        psi_wilt = psi.new_tensor(self.stress_params.wilting_point)
        psi_dry = psi.new_tensor(self.stress_params.optimal_range[0])
        psi_wet = psi.new_tensor(self.stress_params.optimal_range[1])
        psi_anaerobic = psi.new_tensor(self.stress_params.anaerobic_point)

        # References: Feddes et al. (1978); Skaggs et al. (2006), Eq. 13.
        # Mathematical expression: alpha is the four-threshold piecewise reduction.
        # Parameter meanings: psi and all thresholds are soil matric potentials.
        # Units: psi and thresholds [Pa]; alpha [dimensionless].
        # Open-system route: alpha only controls the explicit root-uptake port.
        dry_recovery = (psi - psi_wilt) / (psi_dry - psi_wilt)
        wet_reduction = (psi_anaerobic - psi) / (psi_anaerobic - psi_wet)
        alpha = torch.zeros_like(psi)
        alpha = torch.where(
            (psi > psi_wilt) & (psi < psi_dry), dry_recovery, alpha
        )
        alpha = torch.where(
            (psi >= psi_dry) & (psi <= psi_wet), torch.ones_like(psi), alpha
        )
        alpha = torch.where(
            (psi > psi_wet) & (psi < psi_anaerobic), wet_reduction, alpha
        )
        return torch.clamp(alpha, min=self.zero_clamp, max=self.unit_clamp)
    
    def compute_root_uptake_sink(self,
                                matric_potential: Optional[Tensor] = None,
                                depth_profile: Optional[Tensor] = None,
                                potential_transpiration: Optional[Tensor] = None,
                                root_density: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Compute root water uptake sink term S(z).
        
        Parameters:
            matric_potential : Tensor
                Soil matric potential by layer [Pa]
            depth_profile : Tensor
                Layer bottom depths measured from surface [m]
            potential_transpiration : Tensor
                Potential transpiration demand [kg m-2 s-1]
            root_density : Optional[Tensor]
                Root density profile [kg m-3]
        Returns:
            Tuple[Tensor, Tensor]: 
                - Root uptake sink by layer [kg m-3 s-1]
                - Effective transpiration achieved [kg m-2 s-1]
                
        Mathematical Expression:
            S(z) = alpha(psi(z)) * beta(z) * T_p
            T_actual = integral S(z) dz over root zone
        """
        # Validate required parameters
        if matric_potential is None or depth_profile is None or potential_transpiration is None:
            raise ValueError("Required parameters missing")
        if matric_potential.shape != depth_profile.shape:
            raise ValueError(
                "matric_potential and depth_profile must have the same number of layers"
            )
        if potential_transpiration.numel() != 1:
            raise ValueError("potential_transpiration must be a scalar flux")

        # Compute root distribution
        beta_z = self.compute_root_distribution(depth_profile, root_density)  # m-1
        
        # Compute water stress factor for each layer
        alpha_z = self.compute_water_stress_factor(matric_potential)  # [1]

        # References: Feddes et al. (1978); Skaggs et al. (2006).
        # Mathematical expression: S_i = alpha_i * beta_i * T_p.
        # Parameter meanings: alpha is stress, beta is normalized root density,
        # and T_p is potential transpiration demand.
        # Units: alpha [1], beta [m-1], T_p [kg m-2 s-1], S_i [kg m-3 s-1].
        # Open-system route: integral S_i dz equals effective_transpiration.
        boundaries = torch.cat(
            [depth_profile.new_zeros(1), depth_profile]
        )
        layer_thickness = torch.diff(boundaries)  # m
        
        potential_uptake_density = alpha_z * beta_z * potential_transpiration  # kg m-3 s-1
        
        # Compute total potential uptake
        total_potential = torch.sum(potential_uptake_density * layer_thickness)  # kg m-2 s-1
        
        # Apply water limitation (actual cannot exceed potential demand)
        if torch.any(total_potential > potential_transpiration):
            # Scale down if total exceeds demand
            scaling_factor = potential_transpiration / torch.clamp(total_potential, min=self.positive_epsilon)
            actual_uptake_density = potential_uptake_density * scaling_factor
        else:
            actual_uptake_density = potential_uptake_density
        
        # Compute effective transpiration
        effective_transpiration = torch.sum(actual_uptake_density * layer_thickness)
        
        return actual_uptake_density, effective_transpiration
    
    def compute_root_zone_moisture(self,
                                 soil_moisture: Tensor,
                                 depth_profile: Tensor,
                                 root_density: Optional[Tensor] = None) -> Tensor:
        """
        Compute root-zone weighted average soil moisture.
        
        Parameters:
            soil_moisture : Tensor
                Soil moisture by layer [m3 m-3]
            depth_profile : Tensor
                Depth coordinates [m]
            root_density : Optional[Tensor]
                Root density profile [kg m-3]
                
        Returns:
            Tensor: Root zone moisture [m3 m-3]
        """
        depth_profile = torch.as_tensor(
            depth_profile, device=soil_moisture.device, dtype=soil_moisture.dtype
        )
        if root_density is not None:
            root_density = torch.as_tensor(
                root_density, device=soil_moisture.device, dtype=soil_moisture.dtype
            )
        # Get root distribution weights
        beta_z = self.compute_root_distribution(depth_profile, root_density)
        
        if soil_moisture.shape != depth_profile.shape:
            raise ValueError(
                "soil_moisture and depth_profile must have the same number of layers"
            )
        layer_thickness = torch.diff(
            depth_profile, prepend=depth_profile.new_zeros(1)
        )
        
        # Compute weighted average
        weights = beta_z * layer_thickness
        total_weight = torch.sum(weights)
        if not bool(total_weight > self.positive_epsilon):
            raise ValueError("Root-zone weights must have positive support")
        root_zone_moisture = torch.sum(soil_moisture * weights) / total_weight
        
        return root_zone_moisture
    
    def forward(self,
               state: StateDict,
               time: float,
               dt: float,
               **kwargs) -> FluxDict:
        """
        Forward pass for soil-vegetation coupling.
        
        Computes root water uptake using the Feddes model from soil water
        potential, root distribution, and transpiration demand. The method
        returns coupling ports and observations without advancing soil moisture.
        
        Parameters:
            state : StateDict
                Current system state containing:
                - soil_moisture_profile [m3 m-3] by layer
                - soil_matric_potential [Pa] by layer
                - potential_transpiration [kg m-2 s-1]
                - soil_depth_layers [m] layer-bottom depths (n values, top at 0)
                - root_density_profile [kg m-3] (optional)
            time : float
                Current simulation time [s]
            dt : float
                Time step [s]
            **kwargs
                Additional environment parameters
                
        Returns:
            FluxDict: Output coupling ports and observations
                - root_uptake_profile [kg m-3 s-1]
                - effective_transpiration [kg m-2 s-1]
                - diagnostics.soil_vegetation.feddes_stress_factor [1]
                - diagnostics.soil_vegetation.root_zone_moisture [m3 m-3]
                - diagnostics.soil_vegetation.root_distribution [m-1]
                - diagnostics.soil_vegetation.water_balance_residual [kg m-2 s-1]
        """
        # References: Feddes et al. (1978); Skaggs et al. (2006), Eq. 13; Jackson et
        #   al. (1996); module docstring Mathematical Framework.
        # Mathematical expression: S_i = alpha(psi_i) beta_i T_p, scaled by
        #   T_p / sum_i S_i dz_i when that sum exceeds T_p; T_act = sum_i S_i dz_i;
        #   alpha_bar and theta_rz are beta_i dz_i weighted means; residual = sum_i S_i dz_i - T_act.
        # Parameter meanings: psi_i matric potential; alpha four-threshold Feddes
        #   reduction; beta_i normalized root distribution; T_p potential transpiration;
        #   dz_i thickness from cumulative layer-bottom depths; theta_i soil moisture.
        # Units: theta [m3 m-3]; psi [Pa]; T_p, T_act, residual [kg m-2 s-1]; depths,
        #   dz [m]; root density [kg m-3]; beta [m-1]; alpha [1]; S_i [kg m-3 s-1];
        #   all outputs REPLACE UpdateItems, per_timestep=False; no state advance.
        # Domain/constraints: four required finite 1-D profiles of equal length; theta
        #   in [0, 1]; scalar T_p >= 0; depths positive, strictly increasing; root density
        #   >= 0; dt > 0; water ledger within conservation tolerance else ValueError.
        # Sign convention: S_i >= 0 is a soil-water sink (root extraction) equal to a
        #   positive transpiration flux to the canopy; residual > 0 means uptake > T_act.
        required_inputs = (
            "soil_moisture_profile",
            "soil_matric_potential",
            "potential_transpiration",
            "soil_depth_layers",
        )
        missing_inputs = [key for key in required_inputs if key not in state]
        if missing_inputs:
            raise ValueError(
                "Required state variable missing: " + ", ".join(missing_inputs)
            )

        soil_moisture = torch.as_tensor(state["soil_moisture_profile"])
        if soil_moisture.ndim != 1 or soil_moisture.numel() < 1:
            raise ValueError("soil_moisture_profile must be a non-empty profile")
        target_device = soil_moisture.device
        target_dtype = soil_moisture.dtype
        matric_potential = torch.as_tensor(
            state["soil_matric_potential"], device=target_device, dtype=target_dtype
        )
        depth_layers = torch.as_tensor(
            state["soil_depth_layers"], device=target_device, dtype=target_dtype
        )
        potential_transp = torch.as_tensor(
            state["potential_transpiration"],
            device=target_device,
            dtype=target_dtype,
        )
        root_density = state.get("root_density_profile")
        if root_density is not None:
            root_density = torch.as_tensor(
                root_density, device=target_device, dtype=target_dtype
            )

        profile_tensors = {
            "soil_moisture_profile": soil_moisture,
            "soil_matric_potential": matric_potential,
            "soil_depth_layers": depth_layers,
        }
        if root_density is not None:
            profile_tensors["root_density_profile"] = root_density
        for key, value in profile_tensors.items():
            if value.ndim != 1 or value.shape != soil_moisture.shape:
                raise ValueError(
                    f"{key} and soil_moisture_profile must have the same number of layers"
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"{key} must contain only finite values")
        if potential_transp.numel() != 1:
            raise ValueError("potential_transpiration must be a scalar flux")
        if not bool(torch.all(torch.isfinite(potential_transp))):
            raise ValueError("potential_transpiration must contain only finite values")
        if bool(torch.any(soil_moisture < 0.0)) or bool(
            torch.any(soil_moisture > 1.0)
        ):
            raise ValueError("soil_moisture_profile must be within [0, 1]")
        if bool(potential_transp < 0.0):
            raise ValueError("potential_transpiration must be non-negative")
        if bool(torch.any(depth_layers <= 0.0)) or (
            depth_layers.numel() > 1
            and bool(torch.any(torch.diff(depth_layers) <= 0.0))
        ):
            raise ValueError("soil_depth_layers must be strictly increasing layer bottoms")
        if root_density is not None and bool(torch.any(root_density < 0.0)):
            raise ValueError("root_density_profile must be non-negative")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        
        # Compute layer thickness
        layer_thickness = torch.diff(depth_layers, prepend=depth_layers.new_zeros(1))
        
        # Compute root water uptake
        root_uptake_sink, effective_transpiration = self.compute_root_uptake_sink(
            matric_potential=matric_potential,
            depth_profile=depth_layers,
            potential_transpiration=potential_transp,
            root_density=root_density,
        )
        
        # Compute overall water stress factor
        stress_factors = self.compute_water_stress_factor(matric_potential)
        root_weights = self.compute_root_distribution(depth_layers, root_density)
        
        # Root-weighted average stress
        total_root_weight = torch.sum(root_weights * layer_thickness)
        if not bool(total_root_weight > self.positive_epsilon):
            raise ValueError("Root weights must have positive depth-integrated support")
        overall_stress = (
            torch.sum(stress_factors * root_weights * layer_thickness)
            / total_root_weight
        )
        
        # Compute root zone moisture
        root_zone_moisture = self.compute_root_zone_moisture(
            soil_moisture=soil_moisture,
            depth_profile=depth_layers,
            root_density=root_density
        )
        
        water_balance_residual = (
            torch.sum(root_uptake_sink * layer_thickness)
            - effective_transpiration
        )
        # One column: a 1-D layer profile against a scalar flux, so one ledger (batch_ndim=0).
        conservation_diag = self._conservation_enforcer.check_conservation(
            source_data=root_uptake_sink,
            target_data=effective_transpiration,
            property_name="water_balance",
            source_volume=layer_thickness,
            tolerance=self.conservation_tolerance,
            batch_ndim=0,
        )
        if conservation_diag.relative_error > self.conservation_tolerance:
            raise ValueError(
                "SoilVegetationCoupling water ledger exceeds conservation tolerance"
            )

        def replace(value: Tensor, unit: str) -> UpdateItem:
            return UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                unit=unit,
                per_timestep=False,
                source_functor=self.name,
            )

        outputs = {
            "root_uptake_profile": replace(root_uptake_sink, "kg m-3 s-1"),
            "effective_transpiration": replace(
                effective_transpiration, "kg m-2 s-1"
            ),
            "diagnostics.soil_vegetation.feddes_stress_factor": replace(
                overall_stress, "dimensionless"
            ),
            "diagnostics.soil_vegetation.root_zone_moisture": replace(
                root_zone_moisture, "m3 m-3"
            ),
            "diagnostics.soil_vegetation.root_distribution": replace(
                root_weights, "m-1"
            ),
            "diagnostics.soil_vegetation.water_balance_residual": replace(
                water_balance_residual, "kg m-2 s-1"
            ),
        }

        return outputs
    
    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """
        Run water balance checks using the shared ConservationEnforcer.

        References:
            - Feddes root uptake water balance: integral S(z) dz equals transpiration.
        Mathematical expression:
            |sum(root_uptake_profile) - transpiration| / |transpiration| <= tolerance
        """
        uptake_update = output_updates.get("root_uptake_profile")
        transpiration_update = output_updates.get("effective_transpiration")
        depth_layers = input_state.get("soil_depth_layers")
        if (
            uptake_update is None
            or transpiration_update is None
            or depth_layers is None
        ):
            return {}

        uptake_profile = (
            uptake_update.value
            if isinstance(uptake_update, UpdateItem)
            else torch.as_tensor(uptake_update)
        )
        transpiration = (
            transpiration_update.value
            if isinstance(transpiration_update, UpdateItem)
            else torch.as_tensor(transpiration_update)
        )
        depth_tensor = torch.as_tensor(
            depth_layers,
            device=uptake_profile.device,
            dtype=uptake_profile.dtype,
        )
        if depth_tensor.shape != uptake_profile.shape:
            raise ValueError(
                "soil_depth_layers and root_uptake_profile must have the same number of layers"
            )
        source_volume = torch.diff(
            depth_tensor, prepend=depth_tensor.new_zeros(1)
        )
        transp_tensor = torch.as_tensor(
            transpiration,
            device=uptake_profile.device,
            dtype=uptake_profile.dtype,
        )
        # One column: the layer profile against its transpiration flux, so one ledger (batch_ndim=0).
        diag = self._conservation_enforcer.check_conservation(
            source_data=uptake_profile,
            target_data=transp_tensor,
            property_name="water_balance",
            source_volume=source_volume,
            tolerance=self.conservation_tolerance,
            batch_ndim=0,
        )
        return {
            "water": diag.relative_error,
            "water_abs": diag.absolute_error,
        }

    def diagnose_conservation(self, 
                            input_state: StateDict,
                            output_fluxes: FluxDict) -> 'DiagnosticsDict':
        """
        Diagnose water mass conservation.
        
        Parameters:
            input_state : StateDict
                Input state variables
            output_fluxes : FluxDict
                Computed output fluxes
                
        Returns:
            DiagnosticsDict: Conservation diagnostics as tensors
        """
        diagnostics = {}

        if not output_fluxes:
            raise ValueError("output_fluxes must contain canonical coupling outputs")
        reference_output = next(iter(output_fluxes.values()))
        reference_tensor = (
            reference_output.value
            if isinstance(reference_output, UpdateItem)
            else torch.as_tensor(reference_output)
        )
        device = reference_tensor.device
        dtype = reference_tensor.dtype

        conservation = self.check_conservation_laws(input_state, output_fluxes)
        water_abs = conservation.get("water_abs", 0.0)
        diagnostics["water_mass_balance_residual"] = torch.tensor(water_abs, device=device, dtype=dtype)

        root_uptake_output = output_fluxes.get("root_uptake_profile")
        if root_uptake_output is None:
            raise ValueError("root_uptake_profile is required for conservation diagnostics")
        root_uptake = (
            root_uptake_output.value
            if isinstance(root_uptake_output, UpdateItem)
            else torch.as_tensor(root_uptake_output, device=device, dtype=dtype)
        )
        negative_uptake_violations = torch.sum(root_uptake < 0).to(dtype=dtype)
        diagnostics['negative_uptake_violations'] = negative_uptake_violations

        return diagnostics

__all__ = [
    'SoilVegetationCoupling',
    'RootDistributionParams',
    'SoilWaterStressParams',
]
