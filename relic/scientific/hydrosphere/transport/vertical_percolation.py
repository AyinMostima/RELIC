"""
Hydrosphere Vertical Percolation Transport Functor

This module implements a reduced-order soil-column bottom drainage closure using
field-capacity excess and van Genuchten-Mualem hydraulic conductivity. It emits
one canonical ``drainage_flux`` boundary port and never advances soil-water state.

Mathematical Expression:
    Darcy-Buckingham equation: q = -D(theta)partialtheta/partialz + K(theta)
    where q is water flux [m s-1], D is diffusivity [m^2 s-1], K is hydraulic conductivity [m s-1]
    
    Hydraulic conductivity: K(theta) = K_s(theta/theta_s)^(2b+3)
    where K_s is saturated hydraulic conductivity, theta is volumetric water content, b is pore size index
    
    Drainage flux: F_drain = (max(0, theta - theta_fc) / tau_drain) * (K / Ks) * rho_w * dz
    where theta_fc is field capacity, tau_drain is drainage time constant [s], K/Ks is relative conductivity, dz is layer thickness [m], rho_w is water density [kg m-3]
    
Scientific References:
    - Darcy, H. (1856). Les Fontaines publiques de la ville de Dijon
    - Buckingham, E. (1907). Studies on the movement of soil moisture. USDA Bureau of Soils
    - van Genuchten, M.T. (1980). A closed-form equation for predicting the hydraulic 
      conductivity of unsaturated soils. Soil Sci. Soc. Am. J., 44, 892-898
    - Mualem, Y. (1976). A new model for predicting the hydraulic conductivity of 
      unsaturated porous media. Water Resources Research, 12, 513-522
    - Clapp, R.B. & Hornberger, G.M. (1978). Empirical equations for some soil 
      hydraulic properties. Water Resources Research, 14, 601-604
    - FAO-56: Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation Paper 56

Implementation Features:
    - van Genuchten-Mualem hydraulic functions
    - Field capacity based drainage threshold
    - Gravitational drainage with effective conductivity
    - Temperature effects on water properties
    - One open-boundary mass flux (kg m-2 s-1)
    - Integration with canonical soil-moisture closures
"""

import logging
from typing import Dict, Any, Mapping, Optional, Union
from enum import Enum

import torch
from torch import Tensor

from ....computation.operators.dec.forms import CochainSpaceSpec
from ....computation.operators.dec.hodge import HodgeStarSpec
from ....computation.operators.iphs import PortSpecIPHS
# Import base classes and types
from ....core.functor import (
    TransportMechanismFunctor,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    DependencyDeclaration,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from ....core.abstractions import (
    StateDict,
    FluxDict
)
from ....core.state import UpdateItem, UpdateMode
from ..interface_mapping import HydrosphereInterfaceMixin
from ..solvers.richards_solver import on_device_of
from ....utils.physical_constants import get_physical_constant
from ....utils.soil_hydraulics import (
    effective_saturation,
    van_genuchten_m_parameter,
    van_genuchten_mualem_conductivity,
)

logger = logging.getLogger(__name__)


class HydraulicModel(str, Enum):
    """
    Hydraulic conductivity model enumeration for vertical percolation.

    Different mathematical models for calculating hydraulic conductivity
    as a function of soil water content/pressure.

    References:
        - van Genuchten, M.T. (1980). Closed-form equation for hydraulic conductivity
        - Brooks, R.H. & Corey, A.T. (1964). Hydraulic properties of porous media
        - Clapp, R.B. & Hornberger, G.M. (1978). Empirical equations for soil hydraulic properties
    """
    VAN_GENUCHTEN_MUALEM = "van_genuchten_mualem"
    BROOKS_COREY = "brooks_corey"
    CLAPP_HORNBERGER = "clapp_hornberger"
    SIMPLE_EXPONENTIAL = "simple_exponential"


class VerticalPercolationTransport(HydrosphereInterfaceMixin, TransportMechanismFunctor):
    """
    Vertical Percolation Transport Mechanism Functor
    
    Implements gravitational drainage and vertical water transport in soil using
    the Darcy-Buckingham equation with van Genuchten-Mualem hydraulic functions.
    Calculates one downward bottom-drainage boundary flux.
    
    Transport Formulation:
        Darcy-Buckingham: q = -D(theta)nablatheta + K(theta)nablaH
        Drainage approximation: F_drain = (max(0, theta - theta_fc) / tau_drain) * (K / Ks) * rho_w * dz
        
        Where:
        - q: water flux density [m s-1]
        - D(theta): soil water diffusivity [m^2 s-1]
        - K(theta): hydraulic conductivity [m s-1]
        - H: hydraulic head [m]
        - theta: volumetric water content [m^3 m-3]
        - rho_w: water density [kg m-3]
        - tau_drain: drainage time constant [s]
        - dz: soil layer thickness [m]
        - Ks: saturated hydraulic conductivity [m s-1]
    
    van Genuchten-Mualem Functions:
        Effective saturation: Se = (theta - thetar)/(thetas - thetar)
        Water retention: Se = [1 + (alpha|h|)^n]^(-m)
        Hydraulic conductivity: K(Se) = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2
        
        Where h is pressure head [m], alpha, n, m, l are fitting parameters
    
    Conservation:
        Water mass conservation: partialtheta/partialt + nabla*q = S
        where S represents sources/sinks [s-1]
        
    Attributes:
        device : torch.device
            Computation device (CPU/GPU)
        
    References:
        - van Genuchten, M.T. (1980). A closed-form equation for predicting hydraulic conductivity. 
          Soil Sci. Soc. Am. J., 44, 892-898
        - Mualem, Y. (1976). A new model for predicting hydraulic conductivity. 
          Water Resources Research, 12, 513-522
        - Clapp, R.B. & Hornberger, G.M. (1978). Empirical equations for soil hydraulic properties. 
          Water Resources Research, 14, 601-604
        - FAO-56: Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation Paper 56
    """

    TRANSPORT_TYPE = "vertical_percolation"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    
    def __init__(self,
                 name: str = "vertical_percolation_transport",
                 config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize vertical percolation transport functor.

        Parameters:
            name : str
                Functor instance name
            config : Optional[ConfigDict]
                General functor configuration

        Mathematical Expression:
            Initialization: F_init: Config -> PercolationFunctor
            Parameter space: Theta = {K_s, theta_s, theta_r, theta_fc, alpha, n, ...}
        """
        # Default configuration
        if config is None:
            config = {}
        else:
            config = dict(config)

        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Reduced-order soil-column bottom drainage using field-capacity excess and van Genuchten-Mualem conductivity",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "F_drain = max(0, theta - theta_fc) * (K/Ks) * rho_w * dz / tau; "
                "F_export = F_drain"
            ),
            references=[
                "van Genuchten, M.T. (1980). Closed-form equation for hydraulic conductivity. Soil Sci. Soc. Am. J., 44, 892-898",
                "Mualem, Y. (1976). New model for hydraulic conductivity of unsaturated media. Water Resources Research, 12, 513-522",
                "Darcy, H. (1856). Les Fontaines publiques de la ville de Dijon. Dalmont, Paris",
                "Buckingham, E. (1907). Studies on movement of soil moisture. USDA Bureau of Soils Bulletin 38",
                "FAO-56: Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56"
            ],
            parameters={
                "hydrosphere.transport.vertical_percolation.enable_temperature_effects": (
                    "Apply temperature dependence to drainage viscosity corrections [dimensionless]"
                ),
                "hydrosphere.transport.vertical_percolation.hydraulic_model": (
                    "Primary hydraulic conductivity formulation used by "
                    "VerticalPercolationTransport [dimensionless]"
                ),
                "hydrosphere.soil_moisture.porosity": (
                    "Saturated volumetric water content theta_s of the soil (porosity) [m3 m-3]"
                ),
                "hydrosphere.physical.viscosity_temperature_factor": (
                    "Linearized temperature sensitivity of the liquid water viscosity [K-1]"
                ),
                "hydrosphere.soil_moisture.saturated_hydraulic_conductivity": (
                    "Saturated hydraulic conductivity for Richards equation [m s-1]"
                ),
                "hydrosphere.soil_moisture.residual_water_content": (
                    "Residual water content (minimum water content at very high suction) [m3 m-3]"
                ),
                "hydrosphere.soil_moisture.field_capacity": (
                    "Soil field capacity (maximum water content at gravitational drainage "
                    "threshold) [m3 m-3]"
                ),
                "hydrosphere.soil_moisture.wilting_point": (
                    "Soil wilting point (minimum plant-available water content) [m3 m-3]"
                ),
                "hydrosphere.soil_moisture.alpha_vg": (
                    "Van Genuchten alpha parameter (inverse air entry pressure) [m-1]"
                ),
                "hydrosphere.soil_moisture.n_vg": (
                    "Van Genuchten n parameter (pore size distribution index) [dimensionless]"
                ),
                "hydrosphere.soil_moisture.root_zone_depth": (
                    "Effective rooting depth for water extraction [m]"
                ),
                "hydrosphere.transport.vg_l": (
                    "Mualem tortuosity parameter (l) for van Genuchten model [dimensionless]"
                ),
                "hydrosphere.transport.bc_lambda": (
                    "Brooks-Corey pore size distribution index (lambda) [dimensionless]"
                ),
                "hydrosphere.transport.ch_b_parameter": (
                    "Clapp-Hornberger b parameter for loam soil [dimensionless]"
                ),
                "hydrosphere.transport.drainage_time_constant": (
                    "Characteristic drainage time scale (1 hour default) [s]"
                ),
                "hydrosphere.transport.soil_layer_thickness": (
                    "Effective soil layer thickness for percolation calculations [m]"
                ),
                "hydrosphere.numerical.temperature_bounds": (
                    "Reasonable temperature bounds for hydrosphere calculations [min, max] [K]"
                ),
                "hydrosphere.numerical.saturation_epsilon": (
                    "Small epsilon for effective saturation bounds to avoid numerical issues "
                    "[dimensionless]"
                ),
                "hydrosphere.numerical.effective_saturation_bounds": (
                    "Bounds for effective saturation in van Genuchten and other soil models [min, "
                    "max] [dimensionless]"
                ),
            },
            input_variables=[
                'soil_moisture',
                'soil_temperature',
            ],
            output_variables=[
                'drainage_flux',
                'diagnostics.vertical_percolation.effective_saturation',
                'diagnostics.vertical_percolation.hydraulic_conductivity',
                'diagnostics.vertical_percolation.relative_hydraulic_conductivity',
                'diagnostics.vertical_percolation.field_capacity_excess',
                'diagnostics.vertical_percolation.water_balance_residual',
            ],
            conservation_laws=["vertical_drainage_boundary"],
            spatial_dimensions=1,
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=['soil_moisture', 'groundwater', 'subsurface_runoff'],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
        )

        super().__init__(
            name=name,
            config=config,
            metadata=metadata,
        )

        # Load all parameters from Parameter Manager
        # References: van Genuchten (1980); Mualem (1976); FAO-56
        # Mathematical expression: K(theta) = Ks * Se^l * [1-(1-Se^(1/m))^m]^2
        # Parameter meanings described in docstrings below

        if config:
            logger.warning(
                "VerticalPercolation no longer supports config overrides; "
                "please use ParameterManager vertical_percolation namespace."
            )

        self.enable_temperature_effects = bool(
            self.get_parameter_value(
                'enable_temperature_effects',
                namespace='hydrosphere.transport.vertical_percolation',
                required=True,
            )
        )

        # Physical constants are resolved only through the physical-constant SSOT.
        self.water_density = float(get_physical_constant('water_density'))
        self.reference_temperature = float(
            get_physical_constant('reference_temperature')
        )

        # Soil hydraulic parameters
        ks_raw = self.get_parameter_value(
            'saturated_hydraulic_conductivity',
            namespace='hydrosphere.soil_moisture',
            required=True,
        )
        self.Ks = self._held_parameter(ks_raw)
        self.Ks_value = self._to_scalar_value(ks_raw)

        # theta_s is the soil porosity, the saturated water content Richards reads.
        theta_s_raw = self.get_parameter_value(
            'porosity',
            namespace='hydrosphere.soil_moisture',
            required=True,
        )
        self.theta_s = self._held_parameter(theta_s_raw)
        self.theta_s_value = self._to_scalar_value(theta_s_raw)

        theta_r_raw = self.get_parameter_value(
            'residual_water_content',
            namespace='hydrosphere.soil_moisture',
            required=True,
        )
        self.theta_r = self._held_parameter(theta_r_raw)
        self.theta_r_value = self._to_scalar_value(theta_r_raw)

        theta_fc_raw = self.get_parameter_value(
            'field_capacity',
            namespace='hydrosphere.soil_moisture',
            required=True,
        )
        self.theta_fc = self._held_parameter(theta_fc_raw)
        self.theta_fc_value = self._to_scalar_value(theta_fc_raw)
        self.theta_wp_value = self._to_scalar_value(
            self.get_parameter_value(
                'wilting_point',
                namespace='hydrosphere.soil_moisture',
                required=True,
            )
        )

        # van Genuchten parameters
        vg_alpha_raw = self.get_parameter_value(
            'alpha_vg',
            namespace='hydrosphere.soil_moisture',
            required=True,
        )
        self.vg_alpha = self._held_parameter(vg_alpha_raw)
        self.vg_alpha_value = self._to_scalar_value(vg_alpha_raw)

        vg_n_raw = self.get_parameter_value(
            'n_vg',
            namespace='hydrosphere.soil_moisture',
            required=True,
        )
        self.vg_n = self._held_parameter(vg_n_raw)
        self.vg_n_value = self._to_scalar_value(vg_n_raw)

        # vg_m is a property derived from the held n value at every read; the
        # float is its construction-time value, kept for introspection like
        # the other *_value attributes.
        self.vg_m_value = self._to_scalar_value(self.vg_m)

        vg_l_raw = self.get_parameter_value(
            'vg_l',
            namespace='hydrosphere.transport',
            required=True,
        )
        self.vg_l = self._held_parameter(vg_l_raw)
        self.vg_l_value = self._to_scalar_value(vg_l_raw)

        # Brooks-Corey and Clapp-Hornberger shape parameters. The learnable
        # values are held (bc_lambda, ch_b_parameter) and the closures read
        # them as tensors at every evaluation, so the gradient of the
        # conductivity reaches them; the *_value floats are the
        # construction-time values, kept for introspection.
        self.bc_lambda = self._held_parameter(
            self.get_parameter_value(
                'bc_lambda', namespace='hydrosphere.transport', required=True
            )
        )
        self.bc_lambda_value = self._to_scalar_value(self.bc_lambda)
        self.ch_b_parameter = self._held_parameter(
            self.get_parameter_value(
                'ch_b_parameter', namespace='hydrosphere.transport', required=True
            )
        )
        self.ch_b_parameter_value = self._to_scalar_value(self.ch_b_parameter)

        # Drainage control parameters. The learnable time constant is held
        # (tau_drain) and the drainage law reads it as a tensor at every
        # evaluation, so the gradient of the flux reaches it;
        # drainage_time_constant is the construction-time value used by the
        # validation below and by introspection.
        self.tau_drain = self._held_parameter(
            self.get_parameter_value(
                'drainage_time_constant',
                namespace='hydrosphere.transport',
                required=True,
            )
        )
        self.drainage_time_constant = self._to_scalar_value(self.tau_drain)

        # Layer discretization [will be added to transport.yaml]
        self.soil_layer_thickness = self._to_scalar_value(
            self.get_parameter_value(
                'soil_layer_thickness',
                namespace='hydrosphere.transport',
                required=True,
            )
        )
        self.root_zone_depth = self._to_scalar_value(
            self.get_parameter_value(
                'root_zone_depth',
                namespace='hydrosphere.soil_moisture',
                required=True,
            )
        )

        self._validate_parameters()

        # Temperature effect parameters [from hydrosphere.physical]
        self.viscosity_temp_factor = self._to_scalar_value(
            self.get_parameter_value(
                'viscosity_temperature_factor',
                namespace='hydrosphere.physical',
                required=True,
            )
        )

        # Numerical/tolerance arrays cached for reuse
        self.temperature_bounds = tuple(
            float(value)
            for value in self.get_parameter_value(
                'temperature_bounds', namespace='hydrosphere.numerical', required=True
            )
        )
        self.saturation_epsilon = self._to_scalar_value(
            self.get_parameter_value(
                'saturation_epsilon', namespace='hydrosphere.numerical', required=True
            )
        )
        self.effective_saturation_bounds = tuple(
            float(value)
            for value in self.get_parameter_value(
                'effective_saturation_bounds',
                namespace='hydrosphere.numerical',
                required=True,
            )
        )

        # Hydraulic model selection [will be added to transport.yaml as string]
        hydraulic_model_value = self.get_parameter_value(
            'hydraulic_model',
            namespace='hydrosphere.transport.vertical_percolation',
            required=True,
        )
        resolved_model = str(hydraulic_model_value).lower()
        valid_models = {m.value for m in HydraulicModel}
        if resolved_model not in valid_models:
            raise ValueError(
                f"Unsupported hydraulic model '{resolved_model}'. "
                f"Valid options: {sorted(valid_models)}"
            )
        self.hydraulic_model = resolved_model

        logger.info(
            f"Initialized {self.__class__.__name__} '{name}' "
            f"(model={self.hydraulic_model})"
        )

    def _to_scalar_value(self, value: Any) -> float:
        if isinstance(value, Tensor):
            return float(value.detach())
        return float(value)

    def _held_parameter(self, value: Any) -> Tensor:
        """Return a parameter value as the object this Functor holds.

        A learnable ParameterManager binding is held by reference, unchanged
        (docs/source/architecture_overview.md Sections 4.2 and 4.4.2). A
        converted copy made at construction, which is what a binding sitting
        on another device would give, keeps its construction value after the
        binding changed, carries an autograd graph, and hides the binding from
        held_manager_parameters() and so from a checkpoint. The binding has
        the device of the latest registration of any Functor holding it; the
        kernel reads it on the device of the state (on_device_of or an
        explicit conversion at use). Any other value becomes a tensor on the
        device of this Functor.
        """
        if self.manager_binding_key(value) is not None:
            return value
        return self._ensure_tensor(value)

    @property
    def vg_m(self) -> Tensor:
        """van Genuchten exponent m [1] from the current n value."""
        # References: van Genuchten (1980), Soil Sci. Soc. Am. J. 44, 892-898
        #   (Mualem restriction m = 1 - 1/n); Mualem (1976), Water Resources
        #   Research 12(3), 513-522.
        # Mathematical expression: m = 1 - 1 / n.
        # Parameter meanings: n is the pore-size distribution index
        #   (hydrosphere.soil_moisture.n_vg); m the derived exponent of
        #   K = Ks Se^l [1 - (1 - Se^(1/m))^m]^2.
        # Units: n and m [1].
        # Domain/constraints: n > 1, so 0 < m < 1. n is a learnable binding held
        #   by reference, so m is derived at every read and never stored: it
        #   follows a changed n, and no autograd graph is carried from one call
        #   into the next. The conductivity helper derives the same m from n.
        # Sign convention: none (a dimensionless shape exponent).
        return van_genuchten_m_parameter(self.vg_n)

    @staticmethod
    def _saturation_power(saturation: Tensor, exponent: Tensor) -> Tensor:
        """Se ** e evaluated in float64 and rounded once to the dtype of Se."""
        # References: Brooks and Corey (1964), Hydrology Paper 3, Colorado State
        #   University; Clapp and Hornberger (1978), Water Resources Research
        #   14(4), 601-604; Goldberg (1991), ACM Computing Surveys 23(1), 5-48
        #   (rounding error of floating-point operations).
        # Mathematical expression: p = Se^e, evaluated as
        #   round(pow_float64(Se, e)); a relative error delta of e changes Se^e
        #   by the factor exp(e ln(Se) delta), about 1 + e ln(Se) delta.
        # Parameter meanings: Se is the bounded effective saturation, e the
        #   closure exponent derived from a learnable shape parameter, p the
        #   relative hydraulic conductivity K/Ks, delta the relative rounding
        #   error of e.
        # Units: Se, e, p, delta [1].
        # Domain/constraints: 0 < Se <= 1 (bounded by the caller); e is a
        #   float64 tensor on the device of Se. An exponent rounded to float32
        #   (delta up to 6e-8) changes p by up to |e ln(Se)| delta, several 1e-6
        #   for steep exponents in dry soil, so the power is taken in float64
        #   and rounded once to the dtype of Se. The tensor exponent keeps the
        #   autograd path to the shape parameter.
        # Sign convention: none (a dimensionless ratio in (0, 1]).
        return torch.pow(saturation.to(torch.float64), exponent).to(
            dtype=saturation.dtype
        )

    def _validate_parameters(self) -> None:
        """Validate parameter consistency and physical bounds."""

        def _to_float(value: Union[float, Tensor]) -> float:
            if isinstance(value, Tensor):
                return float(value.detach())
            return float(value)

        theta_s = _to_float(self.theta_s_value)
        theta_r = _to_float(self.theta_r_value)
        theta_wp = _to_float(self.theta_wp_value)
        theta_fc = _to_float(self.theta_fc_value)
        if theta_s <= 0.0 or theta_s > 1.0:
            raise ValueError("Invalid water content hierarchy")
        if not (0.0 <= theta_r <= theta_wp <= theta_fc <= theta_s):
            raise ValueError("Invalid water content hierarchy")

        vg_alpha = _to_float(self.vg_alpha_value)
        vg_n = _to_float(self.vg_n_value)
        if vg_alpha <= 0.0 or vg_n <= 1.0:
            raise ValueError("Invalid van Genuchten parameters")
        if self.Ks_value <= 0.0:
            raise ValueError("Saturated hydraulic conductivity must be positive")
        if self.drainage_time_constant <= 0.0:
            raise ValueError("Drainage time constant must be positive")
        if self.soil_layer_thickness <= 0.0:
            raise ValueError("Soil layer thickness must be positive")
        if self.water_density <= 0.0:
            raise ValueError("Physical constant water_density must be positive")

    def get_dependencies(self) -> DependencyDeclaration:
        """
        Declare no process dependency for the external-state drainage closure.
        
        Returns:
            DependencyDeclaration: Empty process dependency declaration.

        ``soil_moisture`` and ``soil_temperature`` are canonical StateManager
        inputs. Bucket and Richards Functors are mutually exclusive drainage
        closures, while evapotranspiration outputs are not consumed here.
        """
        return self._merge_declared_dependencies({"required": [], "optional": []})

    def _structural_n_dofs(
        self,
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> int:
        """Resolve the structural audit size without changing runtime tensor shape."""

        if n_dofs is not None:
            value = int(n_dofs)
        elif complex_bundle is not None and hasattr(complex_bundle, "n_dofs"):
            value = int(getattr(complex_bundle, "n_dofs"))
        else:
            layer_count = self.root_zone_depth / max(self.soil_layer_thickness, 1.0e-12)
            value = max(1, int(round(layer_count)))
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for vertical percolation.

        References:
          - Darcy (1856), Buckingham (1907), and Richards (1931).
          - Hirani (2003), Discrete Exterior Calculus.
        Mathematical expression:
          theta_dot = -D_h q_h + s_h, q_h = star_K g_h.
        Parameter meanings:
          theta is canonical soil_moisture [m3 m-3]; K is hydraulic
          conductivity [m s-1]; q_h is drainage_flux [kg m-2 s-1].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_vertical_percolation",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="m3 m-3",
                extensivity="intensive",
                physical_domain="hydrosphere",
                semantic_name="soil_moisture",
                orientation_convention="downward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_vertical_percolation",
                degree=1,
                primality="primal",
                n_dofs=n_dofs,
                units="dimensionless",
                extensivity="intensive",
                physical_domain="hydrosphere",
                semantic_name="vertical_hydraulic_gradient",
                orientation_convention="downward_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_vertical_percolation",
                degree=2,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="drainage_flux",
                orientation_convention="downward_positive",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "hydrosphere.transport.vertical_percolation.enable_temperature_effects",
            "hydrosphere.transport.vertical_percolation.hydraulic_model",
            "hydrosphere.soil_moisture.porosity",
            "hydrosphere.physical.viscosity_temperature_factor",
            "hydrosphere.soil_moisture.saturated_hydraulic_conductivity",
            "hydrosphere.soil_moisture.residual_water_content",
            "hydrosphere.soil_moisture.field_capacity",
            "hydrosphere.soil_moisture.wilting_point",
            "hydrosphere.soil_moisture.alpha_vg",
            "hydrosphere.soil_moisture.n_vg",
            "hydrosphere.soil_moisture.root_zone_depth",
            "hydrosphere.transport.vg_l",
            "hydrosphere.transport.bc_lambda",
            "hydrosphere.transport.ch_b_parameter",
            "hydrosphere.transport.drainage_time_constant",
            "hydrosphere.transport.soil_layer_thickness",
            "hydrosphere.numerical.temperature_bounds",
            "hydrosphere.numerical.saturation_epsilon",
            "hydrosphere.numerical.effective_saturation_bounds",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the vertical percolation mobility Hodge contract.

        References:
          - Darcy (1856), hydraulic mobility.
          - Mualem (1976) and van Genuchten (1980), positive relative conductivity.
        Mathematical expression:
          star_K = diag(rho_w * Ks_i), with Ks_i > 0 and rho_w > 0.
        Parameter meanings:
          rho_w is water density [kg m-3], Ks is saturated hydraulic conductivity
          [m s-1]. The field-capacity storage depth dz belongs to the empirical
          boundary-rate closure, not to the Darcy mobility Hodge.
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        mobility_value = (
            self._to_scalar_value(self.Ks)
            * float(self.water_density)
        )
        values = torch.full(
            (n_dofs,),
            mobility_value,
            dtype=torch.float32,
            device=getattr(self, "device", torch.device("cpu")),
        )
        if bool(torch.any(values <= 0.0)):
            raise ValueError("Vertical percolation mobility must be positive.")

        domain = CochainSpaceSpec(
            complex_id="hydrosphere_vertical_percolation",
            degree=1,
            primality="primal",
            n_dofs=n_dofs,
            units="dimensionless",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="vertical_hydraulic_gradient",
            orientation_convention="downward_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_vertical_percolation",
            degree=2,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="drainage_flux",
            orientation_convention="downward_positive",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_vertical_percolation_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=(
                "hydrosphere.soil_moisture.saturated_hydraulic_conductivity",
            ),
            metric_keys=(
                "hydrosphere.transport.soil_layer_thickness",
            ),
            material_law="darcy_buckingham_vertical_percolation",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Dict[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build vertical percolation Hodge terms for IPHS assembly."""

        return (
            self.build_hodge_contract(
                n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
                strict=True,
            ),
        )

    def energy_terms(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return optional energy terms for structural IPHS audits."""

        return {}

    def entropy_terms(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return optional entropy terms for structural IPHS audits."""

        return {}

    def reversible_structure(
        self,
        state: Dict[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for gravitational drainage."""

        return None

    def dissipative_structure(
        self,
        state: Dict[str, Any],
        complex_bundle: Any = None,
    ) -> Dict[str, Any]:
        """Return the positive drainage mobility contribution.

        References:
          - Darcy (1856), Buckingham (1907), van der Schaft and Maschke (1995).
        Mathematical expression:
          M_h = diag(rho_w * Ks_i), sigma = X_h.T M_h X_h >= 0.
        Parameter meanings:
          Ks_i is saturated hydraulic conductivity [m s-1], and rho_w is
          water density [kg m-3].
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"soil_water_storage[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the vertical drainage matter port.

        References:
          - Richards (1931), Darcy-Buckingham boundary flux form.
          - van der Schaft and Maschke (1995), port variables.
        Mathematical expression:
          P_port = e_h * f_h, with e_h = hydraulic head [m] and
          f_h = drainage_flux [kg m-2 s-1].
        Parameter meanings:
          Source and sink hydraulic heads supply the effort pairing.
        """

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_vertical_percolation",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="m",
            extensivity="intensive",
            physical_domain="hydrosphere",
            semantic_name="vertical_drainage_head",
            orientation_convention="downward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_vertical_percolation",
            degree=2,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="vertical_drainage_mass_flux",
            orientation_convention="downward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="vertical_drainage_boundary_flux",
                effort_var="vertical_drainage_head",
                flow_var="drainage_flux",
                effort_space=effort_space,
                flow_space=flow_space,
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
        """Run the canonical forward calculation required by IPHSDECFunctor."""

        return self.forward(dict(state), time, dt)

    def forward(
        self,
        state: StateDict,
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> FluxDict:
        """
        Forward pass for vertical percolation transport calculation.
        
        Compute one gravitational bottom-drainage boundary flux using the
        field-capacity excess closure and van Genuchten-Mualem conductivity.
        
        Parameters:
            state : StateDict
                Required canonical inputs are ``soil_moisture`` [m3 m-3]
                and ``soil_temperature`` [K].
            time : float
                Current simulation time [s]
            dt : float
                Time step [s]
            **kwargs
                Additional context including environment data
                
        Returns:
            FluxDict
                ``drainage_flux`` [kg m-2 s-1] and namespaced diagnostics.
                
        Raises:
            ValueError: If dt or required inputs are missing or invalid.
            
        Mathematical Expression:
            Drainage flux calculation:
            F_drain = (max(0, theta - theta_fc) / tau_drain) * (K / Ks) * rho_w * dz
            
            The output is an open bottom-boundary port. Soil-water state is not
            advanced locally and must be updated by the selected soil-moisture closure.
        """
        del time, kwargs
        dt_tensor = self._ensure_tensor(dt)
        if dt_tensor.numel() != 1 or not bool(torch.isfinite(dt_tensor).all()):
            raise ValueError("VerticalPercolation dt must be one finite scalar")
        if bool(dt_tensor <= 0.0):
            raise ValueError("VerticalPercolation dt must be positive")

        self._validate_percolation_state(state)
        theta = self._get_soil_water_content(state)
        soil_temperature = self._get_soil_temperature(state)

        # References:
        #   - van Genuchten (1980); Mualem (1976).
        # Mathematical expression:
        #   Se = (theta - theta_r) / (theta_s - theta_r);
        #   K = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2.
        # Parameter meanings:
        #   theta, theta_r, theta_s [m3 m-3]; K, Ks [m s-1]; l, m [-].
        saturation = self._calculate_effective_saturation(theta)
        conductivity = self._calculate_hydraulic_conductivity(
            saturation, soil_temperature
        )

        # References:
        #   - Darcy (1856); Buckingham (1907); van Genuchten (1980).
        # Mathematical expression:
        #   F_drain = max(theta-theta_fc, 0) / tau * (K/Ks) * rho_w * dz.
        # Parameter meanings:
        #   theta_fc [m3 m-3], tau [s], rho_w [kg m-3], dz [m].
        drainage_flux = self._calculate_drainage_flux(theta, conductivity)
        relative_conductivity = conductivity / self.Ks.to(
            device=conductivity.device, dtype=conductivity.dtype
        )
        field_capacity_excess = torch.clamp(
            theta
            - self.theta_fc.to(device=theta.device, dtype=theta.dtype),
            min=0.0,
        )

        # Open-system route: every retained drainage candidate is exported through
        # drainage_flux; there is no local state write, lateral split, or overflow.
        water_balance_residual = drainage_flux - drainage_flux
        outputs = {
            "drainage_flux": UpdateItem(
                value=drainage_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "diagnostics.vertical_percolation.effective_saturation": UpdateItem(
                value=saturation,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="dimensionless",
                source_functor=self.name,
            ),
            "diagnostics.vertical_percolation.hydraulic_conductivity": UpdateItem(
                value=conductivity,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="m s-1",
                source_functor=self.name,
            ),
            "diagnostics.vertical_percolation.relative_hydraulic_conductivity": UpdateItem(
                value=relative_conductivity,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="dimensionless",
                source_functor=self.name,
            ),
            "diagnostics.vertical_percolation.field_capacity_excess": UpdateItem(
                value=field_capacity_excess,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="m3 m-3",
                source_functor=self.name,
            ),
            "diagnostics.vertical_percolation.water_balance_residual": UpdateItem(
                value=water_balance_residual,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
        }
        return outputs

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Return the explicit open bottom-drainage boundary residual.

        References:
            Richards (1931); van Genuchten (1980); Mualem (1976).
        Mathematical expression:
            R_w = F_computed - F_exported.
        Parameter meanings:
            F_computed and F_exported are water mass fluxes [kg m-2 s-1].
            The selected soil-moisture owner advances storage; this transport
            Functor verifies that its computed drainage is fully routed.
        """

        residual_key = "diagnostics.vertical_percolation.water_balance_residual"
        drainage_key = "drainage_flux"

        def _output_tensor(key: str) -> Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing conservation output '{key}'")
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, Tensor):
                raise TypeError(f"Conservation output '{key}' must be a Tensor")
            if value.numel() == 0:
                raise ValueError(f"Conservation output '{key}' must not be empty")
            if not value.is_floating_point():
                raise TypeError(
                    f"Conservation output '{key}' must use a floating dtype"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"Conservation output '{key}' must be finite")
            return value

        self._validate_percolation_state(input_state)
        theta = self._get_soil_water_content(input_state)
        temperature = self._get_soil_temperature(input_state)
        saturation = self._calculate_effective_saturation(theta)
        conductivity = self._calculate_hydraulic_conductivity(
            saturation,
            temperature,
        )
        computed_flux = self._calculate_drainage_flux(theta, conductivity)
        exported_flux = _output_tensor(drainage_key)
        declared_residual = _output_tensor(residual_key)
        computed_flux, exported_flux, declared_residual = torch.broadcast_tensors(
            computed_flux,
            exported_flux,
            declared_residual,
        )
        recomputed_residual = computed_flux - exported_flux
        diagnostic_mismatch = declared_residual - recomputed_residual
        magnitude = torch.maximum(
            torch.max(torch.abs(recomputed_residual)),
            torch.max(torch.abs(diagnostic_mismatch)),
        )
        return {"vertical_drainage_boundary": float(magnitude.detach().item())}
    
    def _validate_percolation_state(self, state: StateDict) -> None:
        """Validate state variables required for percolation calculation."""
        required_vars = ['soil_moisture', 'soil_temperature']
        
        for var in required_vars:
            if var not in state:
                raise ValueError(f"Required state variable '{var}' not found for percolation")

            value = state[var]
            if not isinstance(value, Tensor):
                raise TypeError(f"{var} must be a floating-point Tensor")
            if value.numel() == 0:
                raise ValueError(f"{var} must be non-empty")
            if not value.is_floating_point():
                raise TypeError(f"{var} must be a floating-point Tensor")

        theta = state['soil_moisture']
        temperature = state['soil_temperature']
        if not bool(torch.isfinite(theta).all()):
            raise ValueError("soil_moisture must contain only finite values")
        if not bool(torch.isfinite(temperature).all()):
            raise ValueError("soil_temperature must contain only finite values")
        theta_r = self.theta_r.to(device=theta.device, dtype=theta.dtype)
        theta_s = self.theta_s.to(device=theta.device, dtype=theta.dtype)
        if bool(torch.any(theta < theta_r)) or bool(torch.any(theta > theta_s)):
            raise ValueError("soil_moisture is outside the physical range [theta_r, theta_s]")
        if bool(torch.any(temperature < self.temperature_bounds[0])) or bool(
            torch.any(temperature > self.temperature_bounds[1])
        ):
            raise ValueError("soil_temperature is outside the configured physical range")
    
    def _get_soil_water_content(self, state: StateDict) -> Tensor:
        """
        Get soil water content from state.
        
        Returns:
            Tensor: Canonical volumetric soil moisture [m3 m-3].
        """
        return state['soil_moisture']

    def _get_soil_temperature(self, state: StateDict) -> Tensor:
        """
        Get soil temperature for viscosity effects.
        
        Returns:
            Tensor: Canonical soil temperature [K].
        """
        return state['soil_temperature']
    
    # References: van Genuchten (1980).
    # Formula: Se = (theta - theta_r) / (theta_s - theta_r).
    # Parameters: theta (m3 m-3), theta_r (m3 m-3), theta_s (m3 m-3).
    def _calculate_effective_saturation(self, theta: Tensor) -> Tensor:
        """
        Calculate effective saturation from volumetric water content.
        
        Mathematical Expression:
            Se = (theta - thetar) / (thetas - thetar)
            
        Parameters:
            theta : Tensor
                Volumetric water content [m^3 m-3]
                
        Returns:
            Tensor: Effective saturation [dimensionless], clamped to [0, 1]
        """
        # References:
        #   van Genuchten (1980); Mualem (1976); Brooks and Corey (1964); Clapp and Hornberger (1978).
        # Mathematical expression:
        #   Se=(theta-theta_r)/(theta_s-theta_r), with canonical bounds;
        #   VG: K=Ks*Se^l*[1-(1-Se^(1/m))^m]^2, m=1-1/n;
        #   BC: K=Ks*Se^((2+3*lambda_p)/lambda_p);
        #   CH: K=Ks*Se^(2*b+3); retained simple closure: K=Ks*Se^3.
        # Parameter meanings:
        #   theta_r/theta_s are moisture bounds; n,m,l,lambda_p,b are the selected
        #   hydraulic shape controls. The simple_exponential method name currently
        #   executes a cubic power law and must not be described as exp(c*Se).
        # Units:
        #   Se/theta ratios and shape exponents [1]; K/Ks [m s-1].
        # Domain/constraints:
        #   Select one configured conductivity closure; reuse the canonical VG helper
        #   and the existing saturation limits. No second constitutive implementation.
        # Sign convention:
        #   K is a nonnegative transfer magnitude; the boundary orientation is supplied by forward.
        # Calculate effective saturation
        Se = effective_saturation(
            theta,
            self.theta_r,
            self.theta_s,
            epsilon=self.saturation_epsilon,
        )

        return Se
    
    # References: van Genuchten (1980); Mualem (1976); Brooks and Corey (1964); Clapp and Hornberger (1978).
    # Formula: K depends on selected model (see model-specific methods).
    # Parameters: effective_saturation (dimensionless), temperature (K).
    def _calculate_hydraulic_conductivity(self, 
                                        effective_saturation: Tensor,
                                        temperature: Tensor) -> Tensor:
        """
        Calculate hydraulic conductivity using selected hydraulic model.
        
        Parameters:
            effective_saturation : Tensor
                Effective saturation [dimensionless]
            temperature : Tensor
                Soil temperature [K]
                
        Returns:
            Tensor: Hydraulic conductivity [m s-1]
        """
        if self.hydraulic_model == "van_genuchten_mualem":
            K = self._van_genuchten_mualem_conductivity(effective_saturation)
        elif self.hydraulic_model == "brooks_corey":
            K = self._brooks_corey_conductivity(effective_saturation)
        elif self.hydraulic_model == "clapp_hornberger":
            K = self._clapp_hornberger_conductivity(effective_saturation)
        else:  # simple_exponential
            K = self._simple_exponential_conductivity(effective_saturation)

        # Apply temperature correction if enabled
        if self.enable_temperature_effects:
            K = self._apply_temperature_correction(K, temperature)

        if not bool(torch.isfinite(K).all()):
            raise ValueError("Hydraulic conductivity must contain only finite values")
        if bool(torch.any(K < 0.0)):
            raise ValueError("Hydraulic conductivity must be non-negative")

        return K
    
    # References: van Genuchten (1980); Mualem (1976).
    # Formula: K(Se) = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2.
    # Parameters: Ks (m s-1), Se (dimensionless), l (dimensionless), m (dimensionless).
    def _van_genuchten_mualem_conductivity(self, Se: Tensor) -> Tensor:
        """
        Calculate hydraulic conductivity using van Genuchten-Mualem model.
        
        Mathematical Expression:
            K(Se) = Ks * Se^l * [1 - (1 - Se^(1/m))^m]^2
            
        Where:
            Se: effective saturation [dimensionless]
            l: tortuosity parameter (typically 0.5)
            m: van Genuchten parameter m = 1 - 1/n
            
        References:
            - van Genuchten, M.T. (1980). Soil Sci. Soc. Am. J., 44, 892-898
            - Mualem, Y. (1976). Water Resources Research, 12, 513-522
        """
        # References:
        #   van Genuchten (1980); Mualem (1976); Brooks and Corey (1964); Clapp and Hornberger (1978).
        # Mathematical expression:
        #   Se=(theta-theta_r)/(theta_s-theta_r), with canonical bounds;
        #   VG: K=Ks*Se^l*[1-(1-Se^(1/m))^m]^2, m=1-1/n;
        #   BC: K=Ks*Se^((2+3*lambda_p)/lambda_p);
        #   CH: K=Ks*Se^(2*b+3); retained simple closure: K=Ks*Se^3.
        # Parameter meanings:
        #   theta_r/theta_s are moisture bounds; n,m,l,lambda_p,b are the selected
        #   hydraulic shape controls. The simple_exponential method name currently
        #   executes a cubic power law and must not be described as exp(c*Se).
        # Units:
        #   Se/theta ratios and shape exponents [1]; K/Ks [m s-1].
        # Domain/constraints:
        #   Select one configured conductivity closure; reuse the canonical VG helper
        #   and the existing saturation limits. No second constitutive implementation.
        # Sign convention:
        #   K is a nonnegative transfer magnitude; the boundary orientation is supplied by forward.
        return van_genuchten_mualem_conductivity(
            effective_saturation=Se,
            k_saturated=self.Ks,
            n_vg=self.vg_n,
            tortuosity=self.vg_l,
            epsilon=self.saturation_epsilon,
        )
    
    # References: Brooks and Corey (1964).
    # Formula: K(Se) = Ks * Se^((2 + 3*lambda)/lambda).
    # Parameters: Ks (m s-1), Se (dimensionless), lambda (dimensionless).
    def _brooks_corey_conductivity(self, Se: Tensor) -> Tensor:
        """
        Calculate hydraulic conductivity using Brooks-Corey model.
        
        Mathematical Expression:
            K(Se) = Ks * Se^((2+3lambda)/lambda)
            
        Where lambda is the pore size distribution index.
        """
        # References:
        #   van Genuchten (1980); Mualem (1976); Brooks and Corey (1964); Clapp and Hornberger (1978).
        # Mathematical expression:
        #   Se=(theta-theta_r)/(theta_s-theta_r), with canonical bounds;
        #   VG: K=Ks*Se^l*[1-(1-Se^(1/m))^m]^2, m=1-1/n;
        #   BC: K=Ks*Se^((2+3*lambda_p)/lambda_p);
        #   CH: K=Ks*Se^(2*b+3); retained simple closure: K=Ks*Se^3.
        # Parameter meanings:
        #   theta_r/theta_s are moisture bounds; n,m,l,lambda_p,b are the selected
        #   hydraulic shape controls. The simple_exponential method name currently
        #   executes a cubic power law and must not be described as exp(c*Se).
        # Units:
        #   Se/theta ratios and shape exponents [1]; K/Ks [m s-1].
        # Domain/constraints:
        #   Select one configured conductivity closure; reuse the canonical VG helper
        #   and the existing saturation limits. No second constitutive implementation.
        # Sign convention:
        #   K is a nonnegative transfer magnitude; the boundary orientation is supplied by forward.
        Se_safe = torch.clamp(
            Se,
            min=self.effective_saturation_bounds[0],
            max=self.effective_saturation_bounds[1],
        )

        # Brooks-Corey exponent from the current pore-size index, a learnable
        # binding read at use in float64 on the device of Se (the gradient of
        # K reaches lambda_p through the exponent).
        bc_lambda = self.bc_lambda.to(device=Se_safe.device, dtype=torch.float64)
        exponent = (2.0 + 3.0 * bc_lambda) / bc_lambda
        
        K = on_device_of(self.Ks, Se_safe) * self._saturation_power(Se_safe, exponent)
        
        return K
    
    # References: Clapp and Hornberger (1978).
    # Formula: K(Se) = Ks * Se^(2*b + 3).
    # Parameters: Ks (m s-1), Se (dimensionless), b (dimensionless).
    def _clapp_hornberger_conductivity(self, Se: Tensor) -> Tensor:
        """
        Calculate hydraulic conductivity using Clapp-Hornberger model.
        
        Mathematical Expression:
            K(Se) = Ks * Se^(2b + 3)
            
        Where b is the Clapp-Hornberger parameter.
        """
        # References:
        #   van Genuchten (1980); Mualem (1976); Brooks and Corey (1964); Clapp and Hornberger (1978).
        # Mathematical expression:
        #   Se=(theta-theta_r)/(theta_s-theta_r), with canonical bounds;
        #   VG: K=Ks*Se^l*[1-(1-Se^(1/m))^m]^2, m=1-1/n;
        #   BC: K=Ks*Se^((2+3*lambda_p)/lambda_p);
        #   CH: K=Ks*Se^(2*b+3); retained simple closure: K=Ks*Se^3.
        # Parameter meanings:
        #   theta_r/theta_s are moisture bounds; n,m,l,lambda_p,b are the selected
        #   hydraulic shape controls. The simple_exponential method name currently
        #   executes a cubic power law and must not be described as exp(c*Se).
        # Units:
        #   Se/theta ratios and shape exponents [1]; K/Ks [m s-1].
        # Domain/constraints:
        #   Select one configured conductivity closure; reuse the canonical VG helper
        #   and the existing saturation limits. No second constitutive implementation.
        # Sign convention:
        #   K is a nonnegative transfer magnitude; the boundary orientation is supplied by forward.
        Se_safe = torch.clamp(
            Se,
            min=self.effective_saturation_bounds[0],
            max=self.effective_saturation_bounds[1],
        )

        # Clapp-Hornberger exponent from the current b, a learnable binding
        # read at use in float64 on the device of Se (the gradient of K reaches
        # b through the exponent).
        ch_b = self.ch_b_parameter.to(device=Se_safe.device, dtype=torch.float64)
        exponent = 2.0 * ch_b + 3.0
        
        K = on_device_of(self.Ks, Se_safe) * self._saturation_power(Se_safe, exponent)
        
        return K
    
    # References: RELIC reduced-order empirical relationship.
    # Formula: K(Se) = Ks * Se^3.
    # Parameters: Ks (m s-1), Se (dimensionless).
    def _simple_exponential_conductivity(self, Se: Tensor) -> Tensor:
        """
        Calculate hydraulic conductivity using simple exponential model.
        
        Mathematical Expression:
            K(Se) = Ks * Se^3
            
        This is a reduced-order empirical relationship.
        """
        Se_safe = torch.clamp(
            Se,
            min=self.effective_saturation_bounds[0],
            max=self.effective_saturation_bounds[1],
        )
        
        K = on_device_of(self.Ks, Se_safe) * torch.pow(Se_safe, 3.0)
        
        return K
    
    # References: Allen et al. (1998) FAO-56 (temperature dependence in hydrology).
    # Formula: K(T) = K_ref * exp(viscosity_temp_factor * (T - T_ref)).
    # Parameters: K_ref (m s-1), viscosity_temp_factor (K-1), T (K), T_ref (K).
    def _apply_temperature_correction(self, K: Tensor, temperature: Tensor) -> Tensor:
        """
        Apply temperature correction to hydraulic conductivity based on water viscosity.
        
        Mathematical Expression:
            K(T) = K_ref * eta_ref / eta(T)
            
        Where eta is dynamic viscosity, which decreases with temperature.
        
        Approximation: eta(T) ? exp(A/(T-B)) for water viscosity
        """
        T_ref = self.reference_temperature

        # Reduced-order temperature correction (based on water viscosity)
        # Viscosity decreases by ~2.4%/degC near 20degC
        temp_factor = torch.exp(self.viscosity_temp_factor * (temperature - T_ref))
        
        K_corrected = K * temp_factor
        
        return K_corrected
    
    # References: Darcy (1856); Buckingham (1907); van Genuchten (1980); Mualem (1976).
    # Formula: F_drain = (max(0, theta - theta_fc) / tau_drain) * (K / Ks) * rho_w * dz.
    # Parameters: theta (m3 m-3), theta_fc (m3 m-3), tau_drain (s),
    #             K (m s-1), Ks (m s-1), rho_w (kg m-3), dz (m).
    def _calculate_drainage_flux(self,
                               theta: Tensor,
                               hydraulic_conductivity: Tensor) -> Tensor:
        """
        Calculate gravitational drainage flux.
        
        Mathematical Expression:
            F_drain = (max(0, theta - theta_fc) / tau_drain) * (K / Ks) * rho_w * dz
            
        Reduced-order approximation of Darcy-Buckingham equation for drainage.
        """
        # References:
        #   Darcy (1856); Buckingham (1907); van Genuchten (1980); Mualem (1976).
        # Mathematical expression:
        #   Fdrain=rho_w*dz*max(theta-theta_fc,0)/tau_drain*(K(theta,T)/Ks).
        # Parameter meanings:
        #   theta_fc is field capacity; dz is the supported layer thickness; tau is
        #   the drainage response time; K/Ks is relative hydraulic conductivity.
        # Units:
        #   theta [m3 m-3], dz [m], tau [s], K/Ks [1],
        #   rho_w [kg m-3], Fdrain [kg m-2 s-1].
        # Domain/constraints:
        #   Finite canonical theta/T and positive dt; output is a REPLACE boundary
        #   rate, not Fdrain*dt. The open-boundary checker recomputes this expression;
        #   the forward F-F diagnostic alone is not a storage-conservation proof.
        #   theta_fc, Ks and tau are read from the held values at every evaluation
        #   (on the device of theta, tau also in its dtype), never from a
        #   construction-time copy, so the gradient of Fdrain reaches all three.
        # Sign convention:
        #   Positive drainage is downward export across the soil-column bottom.
        # Calculate excess water above field capacity
        excess_water = torch.clamp(
            theta - on_device_of(self.theta_fc, theta), min=0.0
        )

        # Calculate drainage flux using effective conductivity
        # F = K * (excess water fraction) * rho_w / characteristic_time
        layer_thickness = theta.new_tensor(self.soil_layer_thickness)
        water_density = self._ensure_tensor(self.water_density).to(
            device=theta.device, dtype=theta.dtype
        )

        # Drainage time constant approach
        tau_drain = self.tau_drain.to(device=theta.device, dtype=theta.dtype)
        drainage_rate = excess_water / tau_drain

        # Combine with hydraulic conductivity effect
        K_normalized = hydraulic_conductivity / on_device_of(
            self.Ks, hydraulic_conductivity
        )  # Relative conductivity
        effective_rate = drainage_rate * K_normalized

        # Convert to mass flux [kg m-2 s-1]
        drainage_flux = effective_rate * water_density * layer_thickness

        if not bool(torch.isfinite(drainage_flux).all()):
            raise ValueError("drainage_flux must contain only finite values")
        if bool(torch.any(drainage_flux < 0.0)):
            raise ValueError("drainage_flux must be non-negative")

        return drainage_flux
    
    def get_transport_metadata(self) -> Dict[str, Any]:
        """Get transport-specific metadata for introspection."""
        base_metadata = super().get_transport_metadata()
        
        transport_metadata = {
            'transport_config': {
                'hydraulic_model': self.hydraulic_model,
                'saturated_hydraulic_conductivity': self.Ks_value,
                'saturated_water_content': self.theta_s_value,
                'field_capacity': self.theta_fc_value,
                'vg_parameters': {
                    'alpha': self.vg_alpha_value,
                    'n': self.vg_n_value,
                    'm': self.vg_m_value,
                    'l': self.vg_l_value
                },
                'drainage_time_constant': self.drainage_time_constant,
                'soil_layer_thickness': self.soil_layer_thickness,
            }
        }
        
        # Merge with base metadata
        base_metadata.update(transport_metadata)
        return base_metadata

    def __repr__(self) -> str:
        """Concise string representation."""
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"model='{self.hydraulic_model}', "
                f"Ks={self.Ks_value:.2e})")


# Export for module import
__all__ = ['VerticalPercolationTransport', 'HydraulicModel']
