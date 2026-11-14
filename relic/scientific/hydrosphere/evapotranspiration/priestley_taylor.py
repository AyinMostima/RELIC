"""
Priestley-Taylor Evapotranspiration Model for RELIC

This module implements the Priestley-Taylor equation for simplified 
evapotranspiration calculation in the RELIC framework. The model 
provides equilibrium evapotranspiration estimates based on available 
energy and eliminates the need for wind speed and humidity data.

References
----------
- Priestley, C.H.B. and Taylor, R.J. (1972). On the assessment of surface 
  heat flux and evaporation using large-scale parameters. Monthly Weather Review, 
  100(2), 81-92.
- Penman, H.L. (1948). Natural evaporation from open water, bare soil and grass.
  Proceedings of the Royal Society of London, 193(1032), 120-145.
- de Bruin, H.A.R. (1983). A model for the Priestley-Taylor parameter alpha. 
  Journal of Climate and Applied Meteorology, 22(4), 572-578.

Mathematical Expressions
------------------------
Priestley-Taylor equation:
    ET = alpha * (s / (s + gamma)) * (Rn - G) / lambda

Where:
    ET = evapotranspiration rate [kg m-2 s-1]
    alpha = Priestley-Taylor coefficient [dimensionless]
            (canonical literature value is often around 1.26; runtime value is from parameter database)
    s = slope of saturation vapor pressure curve [Pa K-1]
    gamma = psychrometric constant [Pa K-1]
    Rn = net radiation [W m-2]
    G = soil heat flux [W m-2]
    lambda = latent heat of vaporization [J kg-1]

Supporting equations:
    s = saturation_vapor_pressure_slope_coeff * es / (T_c + tetens_coeff_b)^2
    es = tetens_base_pressure * exp(tetens_coeff_a * T_c / (T_c + tetens_coeff_b))
    where T_c = T - kelvin_offset (temperature in Celsius)
    and all physical coefficients are obtained from the physical-constants
    SSOT through get_physical_constant(...)
    gamma = cp * P / (epsilon * lambda)
    
Priestley-Taylor coefficient variations:
    alpha_lit = 1.26 (original literature value for well-watered surfaces)
    alpha = 1.0-1.5 (typical range for vegetation)
    alpha = 0.6-1.0 (dry conditions or water stress)
    runtime alpha is configured by parameter key hydrosphere.evapotranspiration.alpha_pt

Parameters
----------
net_radiation : float
    Net radiation [W m-2]
air_temperature : float
    Air temperature [K]
atmospheric_pressure : float
    Atmospheric pressure [Pa]
alpha_pt : float, optional
    Priestley-Taylor coefficient [dimensionless], configured via
    hydrosphere.evapotranspiration.alpha_pt from the parameter database
"""

from typing import Any, Dict, Mapping, Optional, Union
import torch
from torch import Tensor

from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.core.functor import (
    ConflictResolutionPolicy,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    TransportMechanismFunctor,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.utils.unified_response_functions import validate_temperature_range, validate_positive_values
from relic.core.abstractions import FluxDict, StateDict
from relic.core.state import UpdateItem, UpdateMode
from relic.data.standards.unit_converter import unit_converter
from relic.utils.physical_constants import get_physical_constant
from ..interface_mapping import HydrosphereInterfaceMixin
from ..utils.psychrometrics import (
    compute_psychrometric_constant,
    compute_saturation_vapor_pressure,
    compute_slope_saturation_vapor_pressure,
)


class PriestleyTaylorET(HydrosphereInterfaceMixin, TransportMechanismFunctor):
    """
    Priestley-Taylor Evapotranspiration Model
    
    Implements the simplified Priestley-Taylor equation for calculating 
    evapotranspiration rates based on equilibrium evaporation concepts.
    This model requires fewer meteorological inputs compared to 
    Penman-Monteith and is useful when wind speed and humidity data 
    are limited or uncertain.
    
    The implementation follows CF-1.8 conventions for variable naming
    and units, and provides options for dynamic alpha coefficient
    based on environmental conditions.
    
    Units Convention (CF-1.8 consistent):
        - evapotranspiration_flux: kg m-2 s-1
        - air_temperature: K
        - net_radiation: W m-2
        - atmospheric_pressure: Pa
        - alpha_pt: dimensionless
    
    Attributes
    ----------
    alpha_pt : float
        Priestley-Taylor coefficient [dimensionless]
    soil_heat_fraction : float
        Fraction of net radiation for soil heat flux [dimensionless]
    enable_dynamic_alpha : bool
        Whether to adjust alpha based on environmental conditions
    """

    TRANSPORT_TYPE = "gas_exchange"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    # Declared inputs of the dynamic alpha, selected by config["dynamic_alpha_inputs"] (a
    # list of these names). An absent key declares none and keeps the undeclared route
    # unchanged: theta from state["soil_water_content"] and VPD from the keyword
    # arguments, each applied only when present.
    #   "soil_moisture": volumetric liquid water content theta [m3 m-3] of the soil
    #     store (the soil_moisture state of the soil water model, e.g. the bucket),
    #     declared, required and read instead of soil_water_content;
    #   "vapor_pressure_deficit": atmospheric vapour pressure deficit D [Pa] (the
    #     forcing of that contract), declared, required and read from the state
    #     instead of the keyword argument.
    # A declared input requires hydrosphere.evapotranspiration.enable_dynamic_alpha true.
    DYNAMIC_ALPHA_INPUTS = ("soil_moisture", "vapor_pressure_deficit")
    # Field capacity and wilting point of the relative soil water of the dynamic alpha,
    # selected by config["dynamic_alpha_moisture_frame"]; an absent key keeps the
    # existing frame:
    #   "porosity_fractions": theta_fc = f_fc theta_s and theta_wp = f_wp theta_r
    #     (hydrosphere.physical.field_capacity_fraction and wilting_point_fraction
    #     times hydrosphere.soil_moisture.porosity and residual_water_content);
    #   "bucket_soil_moisture": theta_fc and theta_wp are the owner parameters
    #     hydrosphere.soil_moisture.field_capacity and wilting_point that
    #     BucketSoilMoisture reads, so that the alpha and the store it reads describe
    #     one soil; requires the declared soil_moisture input.
    DYNAMIC_ALPHA_MOISTURE_FRAMES = ("porosity_fractions", "bucket_soil_moisture")
    # Parameters read only under the bucket_soil_moisture frame.
    _BUCKET_FRAME_PARAMETERS = {
        "hydrosphere.soil_moisture.field_capacity": (
            "Field capacity theta_fc of the soil store, the upper reference of the relative "
            "soil water of the dynamic alpha under the bucket_soil_moisture frame [m3 m-3]"
        ),
        "hydrosphere.soil_moisture.wilting_point": (
            "Wilting point theta_wp of the soil store, the lower reference of the relative "
            "soil water of the dynamic alpha under the bucket_soil_moisture frame [m3 m-3]"
        ),
    }
    
    def __init__(self, 
                 name: str = "priestley_taylor_et",
                 config: Optional[Dict] = None,
                 device: Optional[torch.device] = None) -> None:
        """
        Initialize Priestley-Taylor ET model.
        
        Parameters
        ----------
        name : str, optional
            Unique identifier for this functor instance
        config : Dict, optional
            Configuration parameters including alpha coefficient
        device : torch.device, optional
            Torch device for tensor computations
        """
        # The dynamic-alpha inputs and moisture frame are explicit configuration (see
        # DYNAMIC_ALPHA_INPUTS and DYNAMIC_ALPHA_MOISTURE_FRAMES); absent keys keep the
        # undeclared route and the porosity-fraction frame unchanged.
        selection = config or {}
        requested_alpha_inputs = selection.get("dynamic_alpha_inputs", ())
        if (
            isinstance(requested_alpha_inputs, str)
            or not isinstance(requested_alpha_inputs, (list, tuple))
            or any(item not in self.DYNAMIC_ALPHA_INPUTS for item in requested_alpha_inputs)
            or len(set(requested_alpha_inputs)) != len(requested_alpha_inputs)
        ):
            raise ValueError(
                "PriestleyTaylorET config dynamic_alpha_inputs must be a list of distinct "
                f"names from {list(self.DYNAMIC_ALPHA_INPUTS)}, got {requested_alpha_inputs!r}"
            )
        dynamic_alpha_inputs = tuple(
            item for item in self.DYNAMIC_ALPHA_INPUTS if item in requested_alpha_inputs
        )
        dynamic_alpha_moisture_frame = selection.get(
            "dynamic_alpha_moisture_frame", "porosity_fractions"
        )
        if dynamic_alpha_moisture_frame not in self.DYNAMIC_ALPHA_MOISTURE_FRAMES:
            raise ValueError(
                "PriestleyTaylorET config dynamic_alpha_moisture_frame must be one of "
                f"{list(self.DYNAMIC_ALPHA_MOISTURE_FRAMES)}, got {dynamic_alpha_moisture_frame!r}"
            )
        if (
            dynamic_alpha_moisture_frame == "bucket_soil_moisture"
            and "soil_moisture" not in dynamic_alpha_inputs
        ):
            raise ValueError(
                "PriestleyTaylorET dynamic_alpha_moisture_frame bucket_soil_moisture requires "
                "soil_moisture in dynamic_alpha_inputs"
            )
        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Priestley-Taylor evapotranspiration model for equilibrium ET estimation",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "ET = alpha * (s/(s+gamma)) * (Rn-G) / lambda; "
                "q_eq = star_PT A; below 273.15 K the ice option uses s, gamma and "
                "lambda over ice"
            ),
            references=[
                "Priestley, C.H.B. and Taylor, R.J. (1972). Monthly Weather Review, 100(2), 81-92",
                "Penman, H.L. (1948). Proceedings of the Royal Society of London, 193, 120-145",
                "de Bruin, H.A.R. (1983). Journal of Climate and Applied Meteorology, 22(4), 572-578",
                "Murray, F.W. (1967). Journal of Applied Meteorology, 6(1), 203-204"
            ],
            parameters={
                "hydrosphere.soil_moisture.porosity": (
                    "Saturated volumetric water content theta_s of the soil (porosity) [m3 m-3]"
                ),
                "hydrosphere.soil_moisture.residual_water_content": (
                    "Residual volumetric water content theta_r of the soil [m3 m-3]"
                ),
                "hydrosphere.physical.field_capacity_fraction": (
                    "Field capacity as fraction of saturated water content (typical for loam) "
                    "[dimensionless]"
                ),
                "hydrosphere.physical.wilting_point_fraction": (
                    "Wilting point as multiplier of residual water content (typical for loam) "
                    "[dimensionless]"
                ),
                "hydrosphere.physical.vpd_alpha_baseline": (
                    "Baseline scaling factor for Priestley-Taylor alpha reduction as a function of "
                    "vapor pressure deficit [dimensionless]"
                ),
                "hydrosphere.physical.vpd_alpha_sensitivity": (
                    "Sensitivity of Priestley-Taylor alpha to vapor pressure deficit above the "
                    "threshold [kPa-1]"
                ),
                "hydrosphere.physical.vpd_alpha_threshold": (
                    "Vapor pressure deficit threshold beyond which Priestley-Taylor alpha is "
                    "reduced [kPa]"
                ),
                "hydrosphere.physical.atmospheric_pressure_min": (
                    "Minimum plausible surface atmospheric pressure used to guard hydrosphere flux "
                    "calculations [Pa]"
                ),
                "hydrosphere.physical.atmospheric_pressure_max": (
                    "Maximum plausible surface atmospheric pressure used to guard hydrosphere flux "
                    "calculations [Pa]"
                ),
                "hydrosphere.evapotranspiration.alpha_pt": (
                    "Priestley-Taylor equilibrium evapotranspiration coefficient under well-watered"
                    " conditions [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.soil_heat_fraction": (
                    "Fraction of net radiation allocated to soil heat flux for daily "
                    "Priestley-Taylor estimates [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.soil_heat_flux_source": (
                    "Explicit ground-heat source: fraction or signed soil_heat_flux state [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.sub_freezing_phase": (
                    "Phase of saturation vapor pressure and latent heat below 273.15 K: liquid "
                    "or ice [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.enable_dynamic_alpha": (
                    "Enable soil-moisture and VPD-dependent adjustments to the Priestley-Taylor "
                    "alpha coefficient [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.alpha_min": (
                    "Lower bound for Priestley-Taylor alpha under strong atmospheric or soil "
                    "moisture stress [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.alpha_max": (
                    "Upper bound for Priestley-Taylor alpha reflecting advective enhancement of "
                    "evaporation [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.stress_threshold": (
                    "Relative soil moisture threshold at which Priestley-Taylor alpha begins to "
                    "decline [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.hodge_reference_atmospheric_pressure": (
                    "Reference atmospheric pressure of the static Hodge contract [Pa]"
                ),
                "hydrosphere.numerical.temperature_bounds": (
                    "Reasonable temperature bounds for hydrosphere calculations [min, max] [K]"
                ),
                "hydrosphere.numerical.net_radiation_validation_offset": (
                    "Offset added to net radiation for validation to allow negative values [W m-2]"
                ),
                "hydrosphere.numerical.evapotranspiration_flux_clamp_min": (
                    "Minimum evapotranspiration flux (non-negative constraint for physical "
                    "validity) [kg m-2 s-1]"
                ),
                "hydrosphere.numerical.minimum_drainage_flux": (
                    "Minimum threshold for drainage flux calculations [kg m-2 s-1]"
                ),
                "validation.numerical.clamp_min_zero": (
                    "Minimum clamp value for zero-bounded quantities [dimensionless]"
                ),
                "validation.numerical.clamp_max_one": (
                    "Maximum clamp value for unit-bounded quantities [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.hodge_reference_air_temperature": (
                    "Reference air temperature of the static Hodge contract [K]"
                ),
            },
            input_variables=[
                "net_radiation",
                "air_temperature",
                "atmospheric_pressure",
                "soil_heat_flux",
            ],
            output_variables=[
                "evapotranspiration_flux",
                "latent_heat_flux",
                "diagnostics.evapotranspiration",
            ],
            conservation_laws=["energy"],
            spatial_dimensions=0,  # Point-based model
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["soil_moisture", "energy_balance"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            conflict_resolution_policy=ConflictResolutionPolicy.EXCLUSIVE,
        )
        
        # Only the configured dynamic-alpha inputs and frame parameters are declared.
        metadata.input_variables.extend(dynamic_alpha_inputs)
        if dynamic_alpha_moisture_frame == "bucket_soil_moisture":
            metadata.parameters.update(self._BUCKET_FRAME_PARAMETERS)

        # Initialize parent classes
        super().__init__(
            name=name,
            config=config or {},
            metadata=metadata,
            device=device or (torch.device("cpu")),
        )
        self.dynamic_alpha_inputs = dynamic_alpha_inputs
        self.dynamic_alpha_moisture_frame = dynamic_alpha_moisture_frame
        
        # Set TransportMechanism attributes. Energy-limited equilibrium evaporation
        # (latent-heat phase change) has no scalar diffusivity law, so diffusivity is None;
        # no transport parameter is read that no computation uses.
        self.peclet_number = None
        self.velocity_field = None
        
        # Physical constants are resolved only from the physical-constants SSOT.
        self.latent_heat_vaporization = get_physical_constant(
            "latent_heat_vaporization"
        )
        self.specific_heat_air = get_physical_constant("air_specific_heat_cp")
        water_molecular_weight = get_physical_constant("water_molecular_weight")
        air_molecular_weight = get_physical_constant("air_molecular_weight")
        self.molecular_weight_ratio = water_molecular_weight / air_molecular_weight
        self._unit_converter = unit_converter
        
        # References: Priestley and Taylor (1972); FunctorBase parameter-source contract.
        # Mathematical expression: LE=alpha_base*Delta/(Delta+gamma)*(Rn-G)
        #   for fixed alpha before the existing nonnegative projection.
        # Parameter meanings: alpha_base is the explicit registered config candidate,
        #   or the manager binding when no candidate is supplied; Delta/gamma are
        #   psychrometric terms and Rn-G is available surface energy.
        # Units: alpha_base [1]; Delta/gamma [Pa K-1]; LE/Rn/G [W m-2].
        # Domain/constraints: the canonical accessor enforces the unchanged YAML
        #   units, dtype and bounds; trial values do not replace the manager binding.
        # Sign convention: positive LE is outward surface latent heat export.
        self.alpha_pt = self.get_parameter_value(
            "alpha_pt",
            namespace="hydrosphere.evapotranspiration",
            source_order=("config", "manager"), required=True)
        self.soil_heat_flux_source = self.get_parameter_value(
            "soil_heat_flux_source",
            namespace="hydrosphere.evapotranspiration",
            source_order=("config", "manager"), expected_type=str, required=True,
        )
        if self.soil_heat_flux_source not in {"fraction", "state"}:
            raise ValueError("soil_heat_flux_source must be 'fraction' or 'state'")
        self.optional_input_variables = (
            ("soil_heat_flux",) if self.soil_heat_flux_source == "fraction" else ()
        )
        self.soil_heat_fraction = (
            self.get_parameter_value(
                "soil_heat_fraction",
                namespace="hydrosphere.evapotranspiration", required=True,
            ) if self.soil_heat_flux_source == "fraction" else None
        )
        # References: Murray (1967), J. Appl. Meteor. 6, 203-204; Priestley and Taylor (1972).
        # Mathematical expression: L_s = L_v(0 degC) + L_f; the ice branch itself is
        #   stated where it is evaluated in forward().
        # Parameter meanings: sub_freezing_phase selects the phase used below the
        #   freezing point; a_i and b_i are the Tetens-form ice coefficients; L_s is the
        #   latent heat of sublimation; L_v(0 degC) the vaporization intercept; L_f fusion.
        # Units: a_i [1]; b_i [degC]; L_s, L_v, L_f [J kg-1]; the option is categorical.
        # Domain/constraints: option in {liquid, ice}; liquid is the default and leaves
        #   every existing result unchanged; constants come only from the SSOT.
        # Sign convention: latent heats are positive.
        self.sub_freezing_phase = self.get_parameter_value(
            "sub_freezing_phase",
            namespace="hydrosphere.evapotranspiration",
            source_order=("config", "manager"), expected_type=str, required=True,
        )
        if self.sub_freezing_phase not in {"liquid", "ice"}:
            raise ValueError("sub_freezing_phase must be 'liquid' or 'ice'")
        self.enable_dynamic_alpha = bool(
            self.get_parameter_value(
                "enable_dynamic_alpha",
                namespace="hydrosphere.evapotranspiration", required=True)
        )
        self.alpha_min = self.get_parameter_value(
            "alpha_min",
            namespace="hydrosphere.evapotranspiration", required=True)
        self.alpha_max = self.get_parameter_value(
            "alpha_max",
            namespace="hydrosphere.evapotranspiration", required=True)
        self.stress_threshold = self.get_parameter_value(
            "stress_threshold",
            namespace="hydrosphere.evapotranspiration", required=True)
        self.tetens_base_pressure = get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_base_pressure"
        )
        self.tetens_coeff_a = get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_coefficient_a"
        )
        self.tetens_coeff_b = get_physical_constant(
            "relic.physical.constants.psychrometrics.tetens_coefficient_b"
        )
        self.slope_coefficient = get_physical_constant(
            "relic.physical.constants.psychrometrics."
            "saturation_vapor_pressure_slope_coeff"
        )
        self.celsius_to_kelvin = get_physical_constant("celsius_to_kelvin")
        # The ice-phase constants are read only when the ice branch can be evaluated,
        # so the default Functor reads no constant that no computation uses.
        if self.sub_freezing_phase == "ice":
            self.tetens_ice_coeff_a = get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_ice_coefficient_a"
            )
            self.tetens_ice_coeff_b = get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_ice_coefficient_b"
            )
            self.latent_heat_sublimation = get_physical_constant(
                "relic.physical.constants.water.latent_heat_vaporization_intercept"
            ) + get_physical_constant(
                "relic.physical.constants.water.latent_heat_fusion"
            )
        # The soil's own hydraulic properties (the pair Richards reads): theta_s is
        # the porosity and theta_r the residual water content of the soil.
        self.saturated_water_content = self.get_parameter_value(
            "porosity",
            namespace="hydrosphere.soil_moisture", required=True)
        self.residual_water_content = self.get_parameter_value(
            "residual_water_content",
            namespace="hydrosphere.soil_moisture", required=True)
        self.field_capacity_fraction = self.get_parameter_value(
            "field_capacity_fraction",
            namespace="hydrosphere.physical", required=True)
        self.wilting_point_fraction = self.get_parameter_value(
            "wilting_point_fraction",
            namespace="hydrosphere.physical", required=True)
        self.vpd_alpha_baseline = self.get_parameter_value(
            "vpd_alpha_baseline",
            namespace="hydrosphere.physical", required=True)
        self.vpd_alpha_sensitivity = self.get_parameter_value(
            "vpd_alpha_sensitivity",
            namespace="hydrosphere.physical", required=True)
        self.vpd_alpha_threshold = self.get_parameter_value(
            "vpd_alpha_threshold",
            namespace="hydrosphere.physical", required=True)
        self.alpha_factor_max = self.get_parameter_value(
            "clamp_max_one",
            namespace="validation.numerical",
            required=True,
        )
        # A declared dynamic-alpha input is read only by the dynamic alpha.
        if self.dynamic_alpha_inputs and not self.enable_dynamic_alpha:
            raise ValueError(
                "PriestleyTaylorET config dynamic_alpha_inputs requires "
                "hydrosphere.evapotranspiration.enable_dynamic_alpha true"
            )
        # References: Manabe (1969) bucket; paper-01 composition
        #   C15_soil_water_feedback_on_emission_one_soil (RELICPAPER experiment plan,
        #   registered 2026-10-11), where the retention curve is constructed through the
        #   bucket theta_fc and theta_wp so that every reader of the store describes the
        #   same soil (one-soil frame consistency).
        # Mathematical expression: theta_fc = hydrosphere.soil_moisture.field_capacity,
        #   theta_wp = hydrosphere.soil_moisture.wilting_point, the values that
        #   BucketSoilMoisture reads for its ET limiter, drainage threshold and clip.
        # Units: theta_fc, theta_wp [m3 m-3].
        # Domain/constraints: 0 <= theta_wp < theta_fc, checked here and on every call
        #   (a learnable binding may change after construction), as in the bucket.
        # Sign convention: not applicable (water-content thresholds).
        if self.dynamic_alpha_moisture_frame == "bucket_soil_moisture":
            self.dynamic_alpha_field_capacity = self.get_parameter_value(
                "field_capacity",
                namespace="hydrosphere.soil_moisture", validate=True, required=True)
            self.dynamic_alpha_wilting_point = self.get_parameter_value(
                "wilting_point",
                namespace="hydrosphere.soil_moisture", validate=True, required=True)
            self._check_bucket_frame_order()

    def _check_bucket_frame_order(self) -> None:
        """Require theta_wp < theta_fc of the bucket_soil_moisture frame [m3 m-3]."""
        field_capacity = self._ensure_tensor(self.dynamic_alpha_field_capacity)
        wilting_point = self._ensure_tensor(self.dynamic_alpha_wilting_point)
        if not bool(torch.all(wilting_point < field_capacity)) or bool(
            torch.any(wilting_point < 0.0)
        ):
            raise ValueError(
                "Dynamic-alpha bucket frame requires 0 <= wilting_point < field_capacity, got "
                f"{wilting_point.tolist()} and {field_capacity.tolist()}"
            )
        
    @staticmethod
    def _scalar_from_value(value: Union[float, int, Tensor]) -> float:
        """Convert numeric values to Python float without tensor.item/cpu APIs."""
        if isinstance(value, Tensor):
            detached = value.detach().reshape(-1)
            if detached.numel() == 0:
                raise ValueError("Cannot convert empty tensor to scalar.")
            return detached[:1].to(dtype=torch.float64).tolist()[0]
        return float(value)

    def _structural_n_dofs(
        self,
        n_dofs: Optional[int] = None,
        complex_bundle: Any = None,
    ) -> int:
        """Resolve structural audit size without changing runtime tensor shape."""

        if n_dofs is not None:
            value = int(n_dofs)
        elif complex_bundle is not None and hasattr(complex_bundle, "n_dofs"):
            value = int(getattr(complex_bundle, "n_dofs"))
        else:
            value = 1
        if value < 1:
            raise ValueError("Structural degrees of freedom must be positive.")
        return value

    def declared_forms(self) -> tuple[CochainSpaceSpec, ...]:
        """Declare canonical cochain spaces for Priestley-Taylor ET.

        References:
          - Priestley and Taylor (1972), Penman (1948), Hirani (2003).
        Mathematical expression:
          q_eq = star_PT A, A = Rn - G, L_E = lambda q_eq.
        Parameter meanings:
          q_eq is equilibrium ET [kg m-2 s-1], A is available energy
          [W m-2], and L_E is latent heat flux [W m-2].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_priestley_taylor_et",
                degree=1,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="equilibrium_evapotranspiration_flux",
                orientation_convention="outward_positive",
                boundary_role="port",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_priestley_taylor_et",
                degree=1,
                primality="dual",
                n_dofs=n_dofs,
                units="W m-2",
                extensivity="integral",
                physical_domain="surface_energy_balance",
                semantic_name="latent_heat_flux",
                orientation_convention="outward_positive",
                boundary_role="port",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_priestley_taylor_et",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="W m-2",
                extensivity="intensive",
                physical_domain="surface_energy_balance",
                semantic_name="available_energy",
                orientation_convention="surface_available_positive",
                boundary_role="interior",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        keys = (
            "hydrosphere.soil_moisture.porosity",
            "hydrosphere.soil_moisture.residual_water_content",
            "hydrosphere.physical.field_capacity_fraction",
            "hydrosphere.physical.wilting_point_fraction",
            "hydrosphere.physical.vpd_alpha_baseline",
            "hydrosphere.physical.vpd_alpha_sensitivity",
            "hydrosphere.physical.vpd_alpha_threshold",
            "hydrosphere.physical.atmospheric_pressure_min",
            "hydrosphere.physical.atmospheric_pressure_max",
            "hydrosphere.evapotranspiration.alpha_pt",
            "hydrosphere.evapotranspiration.soil_heat_fraction",
            "hydrosphere.evapotranspiration.soil_heat_flux_source",
            "hydrosphere.evapotranspiration.sub_freezing_phase",
            "hydrosphere.evapotranspiration.enable_dynamic_alpha",
            "hydrosphere.evapotranspiration.alpha_min",
            "hydrosphere.evapotranspiration.alpha_max",
            "hydrosphere.evapotranspiration.stress_threshold",
            "hydrosphere.evapotranspiration.hodge_reference_atmospheric_pressure",
            "hydrosphere.numerical.temperature_bounds",
            "hydrosphere.numerical.net_radiation_validation_offset",
            "hydrosphere.numerical.evapotranspiration_flux_clamp_min",
            "hydrosphere.numerical.minimum_drainage_flux",
            "validation.numerical.clamp_min_zero",
            "validation.numerical.clamp_max_one",
            "hydrosphere.evapotranspiration.hodge_reference_air_temperature",
        )
        if getattr(self, "dynamic_alpha_moisture_frame", None) == "bucket_soil_moisture":
            keys = keys + tuple(self._BUCKET_FRAME_PARAMETERS)
        return keys

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the energy-limited ET Hodge contract.

        References:
          - Priestley and Taylor (1972), Penman (1948), and van der Schaft and Maschke (1995).
          - Allen et al. (1998), FAO-56, Eqs. 8 and 13 (gamma and Delta).
        Mathematical expression:
          star_PT = diag(alpha * Delta / ((Delta + gamma) * lambda)) at the reference state
          (T_ref, p_ref).
        Parameter meanings:
          alpha is the Priestley-Taylor coefficient, Delta the slope of the saturation
          vapour pressure curve at T_ref, gamma the psychrometric constant at p_ref,
          lambda the latent heat of vaporization; T_ref and p_ref are the reference
          state hydrosphere.evapotranspiration.hodge_reference_air_temperature and
          hodge_reference_atmospheric_pressure (never forward inputs).
        Units:
          alpha [1]; Delta, gamma [Pa K-1]; lambda [J kg-1]; T_ref [K]; p_ref [Pa];
          star_PT [kg J-1].
        Domain/constraints:
          T_ref and p_ref lie in their YAML bounds, so Tc + b > 0 gives Delta > 0 and
          p_ref > 0 gives gamma > 0; lambda > 0 is a physical constant. Delta + gamma and
          lambda are therefore positive and are checked, not clamped by the dimensionless
          validation.numerical.epsilon_divide; a non-finite or non-positive value raises.
        Sign convention:
          star_PT maps available energy (positive into the surface) to the outward
          evaporative mass flux.
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        reference_temperature = self._ensure_tensor(
            self.get_parameter_value(
                "hodge_reference_air_temperature",
                namespace="hydrosphere.evapotranspiration",
                required=True,
            )
        )
        reference_pressure = self._ensure_tensor(
            self.get_parameter_value(
                "hodge_reference_atmospheric_pressure",
                namespace="hydrosphere.evapotranspiration",
                required=True,
            )
        )
        saturation_vapor_pressure = compute_saturation_vapor_pressure(
            reference_temperature,
            tetens_base_pressure=self.tetens_base_pressure,
            tetens_coeff_a=self.tetens_coeff_a,
            tetens_coeff_b=self.tetens_coeff_b,
            kelvin_offset=self.celsius_to_kelvin,
        )
        slope_svp = compute_slope_saturation_vapor_pressure(
            reference_temperature,
            es_pa=saturation_vapor_pressure,
            tetens_coeff_b=self.tetens_coeff_b,
            tetens_slope_coeff=self.slope_coefficient,
            kelvin_offset=self.celsius_to_kelvin,
        )
        psychrometric_constant = compute_psychrometric_constant(
            reference_pressure,
            specific_heat_air=self.specific_heat_air,
            molecular_weight_ratio=self.molecular_weight_ratio,
            latent_heat=self.latent_heat_vaporization,
        )
        denominator = slope_svp + psychrometric_constant
        latent_heat = self._scalar_from_value(self.latent_heat_vaporization)
        if not (
            bool(torch.all(torch.isfinite(denominator) & (denominator > 0.0)))
            and latent_heat > 0.0
        ):
            raise ValueError(
                "Priestley-Taylor Hodge denominator Delta + gamma and latent heat must be "
                "finite and positive at the reference state"
            )
        equilibrium_fraction = slope_svp / denominator
        mobility_value = (
            self._scalar_from_value(self.alpha_pt)
            * self._scalar_from_value(equilibrium_fraction)
            / latent_heat
        )
        values = torch.full(
            (n_dofs,),
            mobility_value,
            dtype=torch.float32,
            device=getattr(self, "device", torch.device("cpu")),
        )
        if bool(torch.any(values <= 0.0)):
            raise ValueError("Priestley-Taylor energy mobility must be positive.")

        domain = CochainSpaceSpec(
            complex_id="hydrosphere_priestley_taylor_et",
            degree=0,
            primality="primal",
            n_dofs=n_dofs,
            units="W m-2",
            extensivity="intensive",
            physical_domain="surface_energy_balance",
            semantic_name="available_energy_effort",
            orientation_convention="surface_available_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_priestley_taylor_et",
            degree=1,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="equilibrium_evapotranspiration_mass_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_priestley_taylor_energy_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=(
                "hydrosphere.evapotranspiration.alpha_pt",
                "relic.physical.constants.water.latent_heat_vaporization",
                "relic.physical.constants.air.specific_heat_cp",
                "relic.physical.constants.water.molecular_weight",
                "relic.physical.constants.air.molecular_weight_dry",
                "relic.physical.constants.psychrometrics.tetens_base_pressure",
                "relic.physical.constants.psychrometrics.tetens_coefficient_a",
                "relic.physical.constants.psychrometrics.tetens_coefficient_b",
                "relic.physical.constants.psychrometrics."
                "saturation_vapor_pressure_slope_coeff",
                "relic.physical.constants.temperature.celsius_to_kelvin",
            ),
            metric_keys=(
                "hydrosphere.evapotranspiration.hodge_reference_air_temperature",
                "hydrosphere.evapotranspiration.hodge_reference_atmospheric_pressure",
            ),
            material_law="priestley_taylor_equilibrium_energy_mobility",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build Priestley-Taylor Hodge terms for IPHS assembly."""

        return (
            self.build_hodge_contract(
                n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
                strict=True,
            ),
        )

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return energy terms for structural IPHS audits."""

        return {
            "available_energy": "R_n - G",
            "latent_heat_flux": "lambda * ET",
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return entropy terms for structural IPHS audits."""

        return {
            "energy_limited_dissipation": "sigma = A.T star_PT A / T >= 0",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for scalar equilibrium ET."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive energy-limited ET mobility contribution.

        References:
          - Priestley and Taylor (1972), de Bruin (1983), and van der Schaft and Maschke (1995).
        Mathematical expression:
          M_h = star_PT, sigma = A.T M_h A >= 0.
        Parameter meanings:
          A is available energy [W m-2].
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"available_energy_effort[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare Priestley-Taylor equilibrium ET and latent heat ports."""

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_priestley_taylor_et",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="W m-2",
            extensivity="intensive",
            physical_domain="surface_energy_balance",
            semantic_name="available_energy_effort",
            orientation_convention="surface_available_positive",
            boundary_role="port",
        )
        vapor_flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_priestley_taylor_et",
            degree=1,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="priestley_taylor_equilibrium_mass_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        heat_flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_priestley_taylor_et",
            degree=1,
            primality="dual",
            n_dofs=1,
            units="W m-2",
            extensivity="integral",
            physical_domain="surface_energy_balance",
            semantic_name="priestley_taylor_latent_heat_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="priestley_taylor_equilibrium_flux",
                effort_var="available_energy",
                flow_var="evapotranspiration_flux",
                effort_space=effort_space,
                flow_space=vapor_flow_space,
                power_units="W kg m-4 s-1",
                positive_power_convention="out_of_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="priestley_taylor_latent_heat_flux",
                effort_var="available_energy",
                flow_var="latent_heat_flux",
                effort_space=effort_space,
                flow_space=heat_flow_space,
                power_units="W m-2",
                positive_power_convention="out_of_system",
                port_kind="energy",
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
    
    def forward(self,
        # References:
        #   Priestley and Taylor (1972); Penman (1948).
        # Mathematical expression:
        #   G=fG*Rn (fraction source) or G=state['soil_heat_flux'] (state source);
        #   ET_raw=alpha_eff*Delta/(Delta+gamma)*(Rn-G)/Lv;
        #   ET=max(ET_raw,ET_min); LE=L*ET with L=Lv, or L=Ls in frozen cells of the ice option.
        # Parameter meanings:
        #   alpha_eff is the selected fixed or stress-adjusted coefficient;
        #   Delta is saturation-pressure slope; gamma is the psychrometric constant.
        # Units:
        #   ET [kg m-2 s-1], Rn/G/LE [W m-2], Lv [J kg-1],
        #   Delta/gamma [Pa K-1], alpha_eff/fG [1].
        # Domain/constraints:
        #   Validate configured temperature/pressure ranges; no extra dt multiplication
        #   for output rates. Negative-ET projection is an explicit diagnostic.
        # Sign convention:
        #   Positive ET and LE are outward surface exports; incoming Rn supplies energy.
                state: StateDict,
                time: float,
                dt: float,
                **kwargs) -> FluxDict:
        """
        Compute evapotranspiration using Priestley-Taylor equation.
        
        Parameters
        ----------
        state : StateDict
            Current system state containing environmental variables
        time : float
            Current simulation time [s]
        dt : float  
            Time step [s]
        **kwargs
            Additional context variables (may include stress factors)
            
        Returns
        -------
        FluxDict
            Evapotranspiration flux and diagnostic variables
            
        Raises
        ------
        ValueError
            If required state variables are missing or invalid
        """
        state_view = dict(state)

        # Extract required state variables with validation
        try:
            net_radiation = self._ensure_tensor(state_view["net_radiation"])      # W m-2
            air_temp = self._ensure_tensor(state_view["air_temperature"])         # K
            atm_pressure = self._ensure_tensor(
                state_view["atmospheric_pressure"]
            )  # Pa
        except KeyError as e:
            raise ValueError(f"Required state variable missing for Priestley-Taylor ET: {e}")
        
        # Validate input ranges
        # Get temperature bounds from parameter system
        temp_bounds = self.get_parameter_value(
            "temperature_bounds",
            namespace="hydrosphere.numerical", required=True)
        validate_temperature_range(air_temp, min_temp=temp_bounds[0], max_temp=temp_bounds[1])
        
        net_radiation_offset = self.get_parameter_value(
            "net_radiation_validation_offset",
            namespace="hydrosphere.numerical", required=True)
        validate_positive_values(net_radiation + net_radiation_offset, "net_radiation (with offset)")
        
        # Get pressure bounds from parameter system
        pressure_min = self.get_parameter_value(
            "atmospheric_pressure_min",
            namespace="hydrosphere.physical", required=True)
        pressure_max = self.get_parameter_value(
            "atmospheric_pressure_max",
            namespace="hydrosphere.physical", required=True)
        if torch.any(atm_pressure < pressure_min) or torch.any(atm_pressure > pressure_max):
            raise ValueError(f"Atmospheric pressure outside reasonable range [{pressure_min/1000:.0f}-{pressure_max/1000:.0f} kPa]")
        
        # 1. Compute psychrometric variables
        saturation_vapor_pressure = compute_saturation_vapor_pressure(
            air_temp,
            tetens_base_pressure=self.tetens_base_pressure,
            tetens_coeff_a=self.tetens_coeff_a,
            tetens_coeff_b=self.tetens_coeff_b,
            kelvin_offset=self.celsius_to_kelvin,
        )  # Pa

        slope_svp = compute_slope_saturation_vapor_pressure(
            air_temp,
            es_pa=saturation_vapor_pressure,
            tetens_coeff_b=self.tetens_coeff_b,
            tetens_slope_coeff=self.slope_coefficient,
            kelvin_offset=self.celsius_to_kelvin,
        )  # Pa K-1
        
        psychrometric_constant = compute_psychrometric_constant(
            atm_pressure,
            specific_heat_air=self.specific_heat_air,
            molecular_weight_ratio=self.molecular_weight_ratio,
            latent_heat=self.latent_heat_vaporization,
        )  # Pa K-1

        latent_heat = self._latent_heat_of_phase_change(air_temp)  # J kg-1
        if self.sub_freezing_phase == "ice":
            # References: Murray (1967), J. Appl. Meteor. 6, 203-204 (Tetens form over ice);
            #   Goff and Gratch (1946) as the reference over ice; Priestley and Taylor (1972);
            #   Allen et al. (1998), FAO-56, Eq. 8 (psychrometric constant).
            # Mathematical expression: for T < T_f: e_i = e0*exp(a_i*Tc/(Tc+b_i));
            #   D_i = a_i*b_i*e_i/(Tc+b_i)^2; gamma_i = c_p*P/(eps*L_s); L = L_s;
            #   for T >= T_f the liquid e_s, Delta, gamma and L = L_v are kept unchanged.
            # Parameter meanings: T_f freezing point (the Celsius offset); Tc = T - T_f;
            #   e0 base pressure shared by both phases; a_i, b_i ice coefficients; D_i slope
            #   over ice (the exact derivative of e_i); L_s latent heat of sublimation;
            #   gamma_i psychrometric constant over ice; eps molar-mass ratio water/dry air.
            # Units: T, T_f [K]; Tc, b_i [degC]; e_i, e0, P [Pa]; D_i, gamma_i [Pa K-1];
            #   L_s, L_v [J kg-1]; c_p [J kg-1 K-1]; a_i, eps [1].
            # Domain/constraints: selected cell by cell; admitted temperatures are those of
            #   hydrosphere.numerical.temperature_bounds; the Tetens ice form deviates from
            #   Goff-Gratch by at most 0.3 percent above 253.15 K and 1.7 percent above
            #   233.15 K (2.9 percent at 223.15 K); e_s is continuous at T_f while Delta,
            #   gamma and L change phase, so L_E is discontinuous there; the dynamic-alpha
            #   vapor pressure deficit keeps the liquid form; with a scalar pressure gamma
            #   takes the shape of T;
            #   alpha is not validated for frozen or snow-covered surfaces and this Functor
            #   represents no snow or frozen-soil store.
            # Sign convention: evaporation and sublimation are positive outward.
            frozen = air_temp < self.celsius_to_kelvin
            ice_saturation_vapor_pressure = compute_saturation_vapor_pressure(
                air_temp,
                tetens_base_pressure=self.tetens_base_pressure,
                tetens_coeff_a=self.tetens_ice_coeff_a,
                tetens_coeff_b=self.tetens_ice_coeff_b,
                kelvin_offset=self.celsius_to_kelvin,
            )  # Pa
            ice_slope_svp = compute_slope_saturation_vapor_pressure(
                air_temp,
                es_pa=ice_saturation_vapor_pressure,
                tetens_coeff_b=self.tetens_ice_coeff_b,
                tetens_slope_coeff=self.tetens_ice_coeff_a * self.tetens_ice_coeff_b,
                kelvin_offset=self.celsius_to_kelvin,
            )  # Pa K-1
            ice_psychrometric_constant = compute_psychrometric_constant(
                atm_pressure,
                specific_heat_air=self.specific_heat_air,
                molecular_weight_ratio=self.molecular_weight_ratio,
                latent_heat=self.latent_heat_sublimation,
            )  # Pa K-1
            slope_svp = torch.where(frozen, ice_slope_svp, slope_svp)
            psychrometric_constant = torch.where(
                frozen, ice_psychrometric_constant, psychrometric_constant
            )

        # References: Priestley and Taylor (1972); Allen et al. (1998), FAO-56.
        # Mathematical expression: G=fG*Rn (fraction) or G=G_state (state);
        #   A=Rn-G; residual_A=Rn-G-A. The subsequent ET expression is shared.
        # Parameter meanings: fG prescribed fraction; Rn net radiation; G_state
        #   observed or coupled surface soil heat; A available energy.
        # Units: Rn, G, G_state, A, residual_A [W m-2]; fG [1].
        # Domain/constraints: source is explicit; state G must exist and be
        #   finite. Signed G and negative A are retained, without a fallback.
        # Sign convention: Rn positive into the surface; G positive into soil,
        #   negative G returns heat to the surface. ET is positive outward.
        if self.soil_heat_flux_source == "state":
            if "soil_heat_flux" not in state_view:
                raise ValueError("Required state variable soil_heat_flux is missing")
            soil_heat_flux = self._ensure_tensor(state_view["soil_heat_flux"])
            if not bool(torch.isfinite(soil_heat_flux).all()):
                raise ValueError("soil_heat_flux must contain only finite values")
        else:
            soil_heat_flux = self.soil_heat_fraction * net_radiation
        available_energy = net_radiation - soil_heat_flux         # W m-2
        available_energy_residual = net_radiation - soil_heat_flux - available_energy
        
        # 3. Determine Priestley-Taylor coefficient
        if self.enable_dynamic_alpha:
            alpha_effective = self._compute_dynamic_alpha(state_view, **kwargs)
        else:
            alpha_effective = self.alpha_pt + torch.zeros_like(air_temp)
        
        # References: Priestley and Taylor (1972); Penman (1948); DNDC Dndc_SoilPhysics.cpp.
        # Equation: ET = alpha * (Delta / (Delta + gamma)) * (Rn - G) / lambda.
        # Parameters: alpha Priestley-Taylor coefficient [dimensionless],
        # Delta slope of saturation vapor pressure curve [Pa K-1],
        # gamma psychrometric constant [Pa K-1],
        # Rn net radiation [W m-2], G soil heat flux [W m-2],
        # lambda latent heat of the phase (vaporization; sublimation in frozen cells of the ice option) [J kg-1].
        # 4. Apply Priestley-Taylor equation
        # Equilibrium evaporation fraction
        equilibrium_fraction = slope_svp / (slope_svp + psychrometric_constant)
        
        # Priestley-Taylor evapotranspiration [kg m-2 s-1]
        raw_evapotranspiration_flux = (
            alpha_effective * equilibrium_fraction * available_energy / latent_heat
        )
        zero_floor = self._ensure_tensor(
            self.get_parameter_value(
                "clamp_min_zero",
                namespace="validation.numerical",
                required=True,
            )
        )
        negative_et_projection_residual = torch.clamp(
            -raw_evapotranspiration_flux,
            min=zero_floor,
        )
        
        # Ensure non-negative ET (physical constraint)
        evapotranspiration_flux = torch.clamp(
            raw_evapotranspiration_flux,
            min=self.get_parameter_value(
                "evapotranspiration_flux_clamp_min",
                namespace="hydrosphere.numerical", required=True),
        )
        
        # 5. Compute diagnostic variables
        # References: Priestley and Taylor (1972); the resolved alpha source above.
        # Mathematical expression: ET_pot=max(alpha_base*Delta/(Delta+gamma)*A/Lv,0).
        # Parameter meanings: alpha_base is the same resolved unstressed coefficient;
        #   A is available energy and Lv is the latent heat of the phase (Lv, or Ls in frozen
        #   cells of the ice option).
        # Units: ET_pot [kg m-2 s-1]; alpha_base [1]; A [W m-2]; Lv [J kg-1].
        # Domain/constraints: retain the candidate graph without a manager-first
        #   reread; dynamic stress still affects actual ET through alpha_effective.
        # Sign convention: positive ET_pot is potential outward water-vapor export.
        potential_alpha = self.alpha_pt
        potential_et = torch.clamp(
            potential_alpha * equilibrium_fraction * available_energy / latent_heat,
            min=zero_floor,
        )
        
        # Equilibrium evapotranspiration (alpha = 1.0)
        equilibrium_et = torch.clamp(
            equilibrium_fraction * available_energy / latent_heat,
            min=zero_floor,
        )
        latent_heat_flux = evapotranspiration_flux * latent_heat
        
        # Advection factor (ET/Equilibrium ET)
        advection_factor = torch.where(
            equilibrium_et
            > self.get_parameter_value(
                "minimum_drainage_flux",
                namespace="hydrosphere.numerical", required=True),
            evapotranspiration_flux / equilibrium_et,
            alpha_effective
        )
        
        diagnostics = {
            "potential_evapotranspiration": potential_et,           # kg m-2 s-1
            "equilibrium_evapotranspiration": equilibrium_et,       # kg m-2 s-1
            "alpha_coefficient": alpha_effective,                   # dimensionless
            "equilibrium_fraction": equilibrium_fraction,           # dimensionless
            "advection_factor": advection_factor,                   # dimensionless
            "available_energy": available_energy,                   # W m-2
            "soil_heat_flux": soil_heat_flux,                       # W m-2, downward positive
            "slope_saturation_vapor_pressure": slope_svp,           # Pa K-1
            "psychrometric_constant": psychrometric_constant,       # Pa K-1
            "available_energy_residual": available_energy_residual,  # W m-2
            "negative_et_projection_residual": negative_et_projection_residual,  # kg m-2 s-1
        }

        return {
            "evapotranspiration_flux": self.wrap_update(
                evapotranspiration_flux, mode=UpdateMode.REPLACE
            ),  # kg m-2 s-1 (primary output)
            "latent_heat_flux": self.wrap_update(
                latent_heat_flux, mode=UpdateMode.REPLACE
            ),  # W m-2
            "diagnostics.evapotranspiration": diagnostics,
        }

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Validate the paired water-energy port conversion.

        References:
          - Priestley and Taylor (1972), equilibrium evaporation.
          - Allen et al. (1998), FAO-56 evapotranspiration units.
        Mathematical expression:
          L_E = L * ET, with L the latent heat of each cell's phase.
        Parameter meanings:
          L_E is latent heat flux [W m-2], L is the latent heat of vaporization or,
          below the freezing point under the ice option, of sublimation [J kg-1],
          and ET is water mass flux [kg m-2 s-1].
        Open-system route:
          ET and L_E leave through paired water and energy ports. This check
          validates their conversion only; it does not claim a complete
          surface energy balance without sensible and storage heat fluxes.
        """

        def _require_tensor(name: str) -> Tensor:
            if name not in output_updates:
                raise KeyError(f"Missing conservation output '{name}'")
            value = output_updates[name]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, Tensor):
                raise TypeError(f"Conservation output '{name}' must be a torch.Tensor")
            if value.numel() == 0:
                raise ValueError(f"Conservation output '{name}' must be non-empty")
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(f"Conservation output '{name}' must be finite")
            return value

        evapotranspiration = _require_tensor("evapotranspiration_flux")
        latent_heat = _require_tensor("latent_heat_flux")
        if evapotranspiration.shape != latent_heat.shape:
            raise ValueError(
                "Conservation outputs 'evapotranspiration_flux' and "
                "'latent_heat_flux' must have matching shapes"
            )

        if self.sub_freezing_phase == "ice":
            if "air_temperature" not in input_state:
                raise KeyError(
                    "Missing conservation input 'air_temperature' for the ice option"
                )
            phase_latent_heat = self._latent_heat_of_phase_change(
                self._ensure_tensor(input_state["air_temperature"])
            )
        else:
            phase_latent_heat = self.latent_heat_vaporization
        phase_latent_heat = self._ensure_tensor(phase_latent_heat).to(
            device=evapotranspiration.device, dtype=evapotranspiration.dtype
        )
        expected_latent_heat = evapotranspiration * phase_latent_heat
        absolute_residual = torch.abs(latent_heat - expected_latent_heat)
        scale = torch.maximum(
            torch.abs(latent_heat),
            torch.abs(expected_latent_heat),
        )
        relative_residual = torch.where(
            scale > 0,
            absolute_residual / scale,
            absolute_residual,
        )
        magnitude = torch.max(relative_residual)
        return {"energy": float(magnitude.detach().item())}

    def _compute_dynamic_alpha(self, 
                              state: StateDict, 
                              **kwargs) -> Tensor:
        """
        Compute dynamic Priestley-Taylor coefficient based on environmental conditions.
        
        The alpha coefficient is adjusted based on available stress factors,
        soil moisture, and other environmental conditions that affect the
        departure from equilibrium evaporation.
        
        Parameters
        ----------
        state : StateDict
            Current system state
        **kwargs
            Additional context variables
            
        Returns
        -------
        Tensor
            Dynamic alpha coefficient [dimensionless]
        """
        # References:
        #   Priestley and Taylor (1972) provide the equilibrium evaporation basis.
        #   de Bruin (1983) provides background on interpreting the PT coefficient.
        #   The piecewise soil-moisture and clipped linear VPD factors below are
        #   an author-prescribed RELIC sensitivity formulation. Direct equation
        #   provenance for this exact product has not been verified in those papers.
        # Mathematical expression:
        #   rtheta=clip((theta-theta_wp)/(theta_fc-theta_wp),0,1);
        #   alpha_moist=alpha_min+(alpha_PT-alpha_min)*rtheta/s if rtheta<s, else alpha_PT;
        #   alpha=min(alpha_PT, alpha_moist(rtheta))*clip(b-k*(VPD_kPa-v0),
        #   alpha_min/alpha_PT,f_max); alpha=clip(alpha,alpha_min,alpha_max).
        # Parameter meanings:
        #   theta_fc and theta_wp come from the configured moisture frame: by default
        #   theta_fc = f_fc theta_s and theta_wp = f_wp theta_r, with theta_s the soil
        #   porosity and theta_r its residual water content (as in Richards); under the
        #   bucket_soil_moisture frame the bucket field_capacity and wilting_point;
        #   alpha_moist is the implemented piecewise linear response with s the
        #   stress_threshold; b,k,v0 are the VPD controls and f_max clamp_max_one.
        # Units:
        #   theta, theta_fc, theta_wp [m3 m-3]; rtheta, s and all alpha/factor values
        #   [1]; VPD input [Pa] is converted to [kPa]; k [kPa-1]; v0 [kPa].
        # Domain/constraints:
        #   Apply only the moisture/VPD controls actually supplied. Undeclared route:
        #   theta from state soil_water_content and VPD from kwargs, when present.
        #   Declared route (config dynamic_alpha_inputs): the declared soil_moisture
        #   and vapor_pressure_deficit are required in the state and finite. In a
        #   graph, soil_moisture is the store at the start of the step (an explicit,
        #   start-of-step evaluation). Use the canonical unit conversion.
        # Sign convention:
        #   Stress reduces the empirical evaporation multiplier; this helper writes no water.
        # Start with base alpha
        alpha = self.alpha_pt + torch.zeros_like(
            self._ensure_tensor(state["air_temperature"])
        )
        declared_inputs = getattr(self, "dynamic_alpha_inputs", ())
        moisture_key = (
            "soil_moisture" if "soil_moisture" in declared_inputs else "soil_water_content"
        )
        for key in declared_inputs:
            if state.get(key) is None:
                raise ValueError(
                    f"Required state variable missing for the declared dynamic alpha: {key!r}"
                )
            if not bool(torch.all(torch.isfinite(self._ensure_tensor(state[key])))):
                raise ValueError(f"Declared dynamic-alpha input {key!r} must be finite")
        if "vapor_pressure_deficit" in declared_inputs and "vapor_pressure_deficit" in kwargs:
            raise ValueError(
                "vapor_pressure_deficit is a declared dynamic-alpha input and must not also be "
                "passed as a keyword argument"
            )
        # Adjust based on soil moisture if available
        if moisture_key in state:
            soil_moisture = self._ensure_tensor(state[moisture_key])
            
            # Field capacity and wilting point of the soil: theta_fc = f_fc theta_s and
            # theta_wp = f_wp theta_r.
            field_capacity = self.field_capacity_fraction * self.saturated_water_content
            wilting_point = self.wilting_point_fraction * self.residual_water_content
            if self.dynamic_alpha_moisture_frame == "bucket_soil_moisture":
                # The bucket_soil_moisture frame replaces them by the field capacity and
                # wilting point of the bucket store (see __init__) [m3 m-3], on the device
                # and dtype of theta as the bucket casts its own thresholds.
                self._check_bucket_frame_order()
                field_capacity = self._ensure_tensor(self.dynamic_alpha_field_capacity).to(
                    device=soil_moisture.device, dtype=soil_moisture.dtype
                )
                wilting_point = self._ensure_tensor(self.dynamic_alpha_wilting_point).to(
                    device=soil_moisture.device, dtype=soil_moisture.dtype
                )
            
            # Relative soil moisture
            relative_moisture = torch.clamp(
                (soil_moisture - wilting_point) / (field_capacity - wilting_point),
                min=0.0, max=1.0
            )
            
            # Reduce alpha under water stress
            moisture_factor = torch.where(
                relative_moisture < self.stress_threshold,
                self.alpha_min + (self.alpha_pt - self.alpha_min) * relative_moisture / self.stress_threshold,
                self.alpha_pt
            )
            
            alpha = torch.minimum(alpha, moisture_factor)
        
        # Adjust based on vapor pressure deficit if available
        vpd_source = (
            state if "vapor_pressure_deficit" in declared_inputs else kwargs
        )
        if "vapor_pressure_deficit" in vpd_source:
            vpd = self._ensure_tensor(vpd_source["vapor_pressure_deficit"])

            # References: SI Brochure, 9th edition, for the pressure conversion.
            # The VPD response is the author-prescribed formulation stated above.
            # Equation: VPD_kPa = convert(VPD_Pa, Pa -> kPa).
            # Parameters: VPD is vapor pressure deficit; alpha terms are
            # ParameterManager-backed response coefficients.
            # Units: VPD_Pa [Pa], VPD_kPa [kPa], alpha [1].
            # Open route: this is a unit conversion and diagnostic response;
            # it does not add or remove water or energy.
            # Reduce alpha under high VPD (atmospheric demand)
            vpd_kpa = self._unit_converter.convert(
                vpd,
                "vapor_pressure_deficit",
                "Pa",
                "kPa",
            )
            baseline_alpha = self.vpd_alpha_baseline
            vpd_sensitivity = self.vpd_alpha_sensitivity
            vpd_threshold = self.vpd_alpha_threshold
            raw_vpd_factor = baseline_alpha - vpd_sensitivity * (
                vpd_kpa - vpd_threshold
            )
            vpd_factor = torch.minimum(
                torch.maximum(raw_vpd_factor, self.alpha_min / self.alpha_pt),
                self._ensure_tensor(self.alpha_factor_max),
            )
            
            alpha = alpha * vpd_factor
        
        # Apply bounds
        alpha = torch.clamp(alpha, min=self.alpha_min, max=self.alpha_max)
        
        return alpha
    
    def _latent_heat_of_phase_change(self, air_temp: Tensor) -> float | Tensor:
        """Return the latent heat that pairs the water and energy ports at each cell.

        References:
          - Priestley and Taylor (1972); Allen et al. (1998), FAO-56.
          - Murray (1967) for the phase boundary of the ice option.
        Mathematical expression:
          L = L_s where T < T_f and sub_freezing_phase = ice, otherwise L = L_v,
          with L_s = L_v(0 degC) + L_f; the port pair is L_E = L * ET.
        Parameter meanings:
          T air temperature; T_f freezing point (the Celsius offset); L_v latent heat
          of vaporization; L_s latent heat of sublimation; L_f latent heat of fusion.
        Units:
          T, T_f [K]; L, L_v, L_s, L_f [J kg-1]; L_E [W m-2]; ET [kg m-2 s-1].
        Domain/constraints:
          The default option returns the unchanged scalar L_v; the ice option returns
          one float32 value per cell with the shape of T.
        Sign convention:
          Latent heats are positive; ET and L_E are positive outward.
        """
        if self.sub_freezing_phase != "ice":
            return self.latent_heat_vaporization
        return torch.where(
            air_temp < self.celsius_to_kelvin,
            self._ensure_tensor(self.latent_heat_sublimation),
            self._ensure_tensor(self.latent_heat_vaporization),
        )

    def _ensure_tensor(self, value: Union[float, Tensor]) -> Tensor:
        """
        Ensure input is a tensor with proper device and dtype.
        
        Parameters
        ----------
        value : Union[float, Tensor]
            Input value to convert
            
        Returns
        -------
        Tensor
            Tensor representation of input
        """
        # Preserve double precision and the existing autograd path, with a float32 floor.
        # Scalars, integer tensors and lower-precision floats retain their original conversion.
        dtype = torch.promote_types(value.dtype, torch.float32) if isinstance(value, Tensor) and value.is_floating_point() else torch.float32
        return super()._ensure_tensor(value, dtype=dtype)

# Export the main class
__all__ = ["PriestleyTaylorET"]

