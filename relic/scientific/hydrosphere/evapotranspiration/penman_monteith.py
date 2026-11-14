"""
Penman-Monteith Evapotranspiration Model for RELIC

This module implements the Penman-Monteith equation for evapotranspiration
calculation in the RELIC framework. The implementation follows the 
unified interface requirements and integrates with the FunctorBase system.

References
----------
- Monteith, J.L. (1965). Evaporation and environment. Symposia of the Society 
  for Experimental Biology, 19, 205-224.
- Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998). Crop evapotranspiration - 
  Guidelines for computing crop water requirements. FAO Irrigation and drainage paper 56.
- Penman, H.L. (1948). Natural evaporation from open water, bare soil and grass.
  Proceedings of the Royal Society of London, 193(1032), 120-145.

Mathematical Expressions
------------------------
Penman-Monteith equation:
    ET = [s(Rn - G) + rho * cp * VPD * ga] / [lambda(s + gamma(1 + ga/gc))]

Where:
    ET = evapotranspiration rate [kg m-2 s-1]
    s = slope of saturation vapor pressure curve [Pa K-1]
    Rn = net radiation [W m-2]
    G = soil heat flux [W m-2]
    rho = air density [kg m-3]
    cp = specific heat of air [J kg-1 K-1]
    VPD = vapor pressure deficit [Pa]
    ga = aerodynamic conductance [m s-1]
    lambda = latent heat of vaporization [J kg-1]
    gamma = psychrometric constant [Pa K-1]
    gc = canopy conductance [m s-1]

Supporting equations:
    s = 4098 * es / (T + 237.3)^2  [Pa K-1, T in degC]
    es = 0.6108 * exp(17.27 * T / (T + 237.3))  [kPa, T in degC]
    VPD = es - ea = es * (1 - RH)
    ga = k2 * u / [ln((z - d)/z0m) * ln((z - d)/z0h)]
    gamma = cp * P / (epsilon * lambda)

Parameters
----------
net_radiation : float
    Net radiation [W m-2]
air_temperature : float
    Air temperature [K]
relative_humidity : float
    Relative humidity [0-1]
wind_speed : float
    Wind speed at reference height [m s-1]
atmospheric_pressure : float
    Atmospheric pressure [Pa]
surface_resistance : float
    Surface resistance [s m-1]
aerodynamic_resistance : float, optional
    Aerodynamic resistance [s m-1] - if provided by turbulence module, 
    will be used preferentially over internal calculation to ensure
    single source of truth
"""

from typing import Any, Dict, Mapping, Optional, Union
import math
import torch
from torch import Tensor
import warnings

from ....computation.operators.dec.forms import CochainSpaceSpec
from ....computation.operators.dec.hodge import HodgeStarSpec
from ....computation.operators.iphs import PortSpecIPHS
from ....core.feature_flags import feature_manager
from ....core.functor import (
    ConflictResolutionPolicy,
    DependencyDeclaration,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    TransportMechanismFunctor,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from ....utils.unified_response_functions import validate_temperature_range, validate_positive_values
from ....core.abstractions import FluxDict, StateDict
from ....core.state import UpdateItem, UpdateMode
from ....data.standards.unit_converter import unit_converter
from ....utils.physical_constants import get_physical_constant
from ...atmosphere import daylight_quadrature
from ..interface_mapping import HydrosphereInterfaceMixin
from ..utils.psychrometrics import (
    compute_moist_air_density,
    compute_psychrometric_constant,
    compute_saturation_vapor_pressure,
    compute_slope_saturation_vapor_pressure,
)


def fao56_aerodynamic_conductance(
    wind_speed: Tensor,
    *,
    reference_height: Any,
    displacement_height: Any,
    roughness_length: Any,
    heat_to_momentum_ratio: Any,
    von_karman_constant: float,
    minimum_wind_speed: Any,
    conductance_bounds: Any,
) -> Tensor:
    """Neutral log-profile aerodynamic conductance g_a of FAO-56 Eq. 4 [m s-1].

    The one RELIC owner of this law: PenmanMonteithET._compute_aerodynamic_conductance
    and LeafEnergyBalance (relic.scientific.biosphere.photosynthesis.leaf_energy_balance)
    call it with the parameters of the hydrosphere.evapotranspiration,
    hydrosphere.physical and hydrosphere.numerical namespaces, so that both read one
    aerodynamic resistance r_a = 1 / g_a for heat and water vapour.
    """
    # References:
    #   Allen et al. (1998), FAO-56, Eq. 4; Monteith and Unsworth (2013).
    # Mathematical expression:
    #   ga = kappa^2*max(u,u_min) / [ln((z-d)/z0m)*ln((z-d)/z0h)], z0h = c_h z0m,
    #   then ga is limited to [ga_min, ga_max].
    # Parameter meanings:
    #   u is reference wind; z-d is effective height; z0m/z0h are roughness lengths;
    #   c_h the heat-to-momentum roughness ratio; kappa is the physical von Karman
    #   constant; u_min the declared wind floor; [ga_min, ga_max] the declared bounds.
    # Units:
    #   ga/u/u_min [m s-1]; heights/roughness [m]; logarithms, c_h and kappa [1].
    # Domain/constraints:
    #   Neutral log-profile approximation only (no stability correction); positive
    #   roughness and effective height above roughness, otherwise ValueError.
    # Sign convention:
    #   Positive ga denotes a transfer magnitude; the caller supplies the gradient sign.
    wind_speed_safe = torch.clamp(wind_speed, min=minimum_wind_speed)  # m s-1

    # Heights above displacement with validation
    effective_height = reference_height - displacement_height
    if effective_height <= 0:
        raise ValueError(f"Effective height (z-d) must be positive: {effective_height}")

    momentum_roughness = roughness_length
    heat_roughness = momentum_roughness * heat_to_momentum_ratio  # Typical ratio for heat to momentum roughness

    # Validate roughness lengths to prevent log(<=0)
    if momentum_roughness <= 0 or heat_roughness <= 0:
        raise ValueError(f"Roughness lengths must be positive: z0m={momentum_roughness}, z0h={heat_roughness}")

    # Additional check: effective height must be greater than roughness
    if effective_height <= momentum_roughness:
        raise ValueError(f"Effective height ({effective_height}) must exceed roughness length ({momentum_roughness})")

    # Convert to tensor for computation
    effective_height_t = torch.as_tensor(
        effective_height, dtype=wind_speed.dtype, device=wind_speed.device
    )
    momentum_roughness_t = torch.as_tensor(
        momentum_roughness, dtype=wind_speed.dtype, device=wind_speed.device
    )
    heat_roughness_t = torch.as_tensor(
        heat_roughness, dtype=wind_speed.dtype, device=wind_speed.device
    )

    # Logarithmic terms with guaranteed positive arguments
    momentum_term = torch.log(effective_height_t / momentum_roughness_t)
    heat_term = torch.log(effective_height_t / heat_roughness_t)

    # Aerodynamic conductance (neutral stability approximation)
    ga = (von_karman_constant**2 * wind_speed_safe) / (momentum_term * heat_term)

    # Apply reasonable bounds
    ga = torch.clamp(ga, min=conductance_bounds[0], max=conductance_bounds[1])  # m s-1

    return ga


class PenmanMonteithET(HydrosphereInterfaceMixin, TransportMechanismFunctor):
    """
    Penman-Monteith Evapotranspiration Model
    
    Implements the standard Penman-Monteith equation for calculating 
    evapotranspiration rates based on energy balance and aerodynamic 
    principles. This model provides physically-based ET estimates that
    account for both energy and vapor pressure limitations.
    
    The implementation follows CF-1.8 conventions for variable naming
    and units, and integrates with the unified response function library
    for temperature dependencies.
    
    Units Convention (CF-1.8 conformant):
        - evapotranspiration_flux: kg m-2 s-1
        - air_temperature: K
        - net_radiation: W m-2
        - wind_speed: m s-1
        - atmospheric_pressure: Pa
        - relative_humidity: dimensionless (0-1)
    
    Attributes
    ----------
    roughness_length : float
        Momentum roughness length [m]
    displacement_height : float
        Zero plane displacement height [m]
    minimum_conductance : float
        Minimum canopy conductance to prevent division by zero [m s-1]
    """

    TRANSPORT_TYPE = "gas_exchange"

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"
    # Declared canopy (surface) resistance sources; config["canopy_resistance_source"]
    # must name one of them, there is no implicit default:
    #   "state": r_s from the surface_resistance state [s m-1], or r_s = 1 / g_c from the
    #     canopy_conductance state [m s-1] when surface_resistance is absent.
    #   "fao56_reference_surface": r_s of the FAO-56 hypothetical reference grass
    #     (Allen et al. 1998), read from hydrosphere.physical.reference_surface_resistance.
    CANOPY_RESISTANCE_SOURCES = ("state", "fao56_reference_surface")
    # State inputs declared (and read) only with canopy_resistance_source "state".
    STATE_RESISTANCE_INPUTS = ("surface_resistance", "canopy_conductance")
    # Declared temporal supports; config["temporal_support"] selects one, and an absent
    # key keeps the hourly formulation unchanged:
    #   "hourly": FAO-56 hourly form (Allen et al. 1998, Eqs. 45-46 for G), with
    #     dt <= hydrosphere.physical.soil_heat_flux_hourly_max_step.
    #   "daily": FAO-56 24-hour form (Allen et al. 1998, Chapter 3: Eq. 12 for e_s and
    #     Eq. 42 for G), qualified for dt = relic.physical.constants.time.seconds_per_day
    #     only; every input is the 24-hour value of its quantity (daily mean rates).
    #   "daylight_gauss": daily step only (dt = seconds_per_day). The law is evaluated at
    #     the n Gauss instants of the daylight period and once for the night of
    #     relic.scientific.atmosphere.daylight_quadrature and averaged with the sample
    #     weights (_daylight_gauss_law): R_n,k = R_n - S_n + S_n,k, T_k and
    #     D_k = max(e0(T_k) - e_a, 0) of the daily course, the per-sample canopy
    #     conductance canopy_conductance_daylight_samples (canopy_resistance_source
    #     "state"), G_k of FAO-56 Eqs. 45-46 (or G_state), wind constant over the day.
    #     actual_vapor_pressure replaces the relative_humidity / vapor_pressure_deficit
    #     inputs. config["daylight_gauss_nodes"] (3 or 5) and
    #     config["daylight_shortwave_shape"] select the rule. Outputs are 24-hour means.
    TEMPORAL_SUPPORTS = ("hourly", "daily", daylight_quadrature.DAYLIGHT_GAUSS)
    # Forcing inputs declared (and required) only with temporal_support "daily": the
    # daily maximum and minimum air temperature [K] of FAO-56 Eq. 12.
    DAILY_TEMPERATURE_INPUTS = ("air_temperature_max", "air_temperature_min")
    # Inputs read (and required) only with temporal_support "daylight_gauss": the daily
    #   maximum and minimum air temperature [K], net shortwave S_n [W m-2] of the radiation
    #   owner, latitude [deg N], day of year [1] and actual vapour pressure e_a [Pa].
    DAYLIGHT_INPUTS = ("air_temperature_max", "air_temperature_min", "shortwave_net",
                       "latitude", "day_of_year", "actual_vapor_pressure")
    # The law inputs the daylight support puts on the sample axis (all required).
    DAYLIGHT_LAW_INPUTS = ("net_radiation", "air_temperature", "wind_speed", "atmospheric_pressure",
                           *DAYLIGHT_INPUTS)
    # The per-sample canopy conductance [m s-1] of a daylight_gauss Medlyn upscaling; it
    #   replaces surface_resistance / canopy_conductance with canopy_resistance_source "state".
    DAYLIGHT_CONDUCTANCE_INPUT = "canopy_conductance_daylight_samples"

    def __init__(self,
                 name: str = "penman_monteith_et",
                 config: Optional[Dict] = None) -> None:
        """
        Initialize Penman-Monteith ET model.

        Parameters
        ----------
        name : str, optional
            Unique identifier for this functor instance
        config : Dict
            Configuration; ``canopy_resistance_source`` is required and must be one of
            ``CANOPY_RESISTANCE_SOURCES``
        """
        canopy_resistance_source = (config or {}).get("canopy_resistance_source")
        if canopy_resistance_source not in self.CANOPY_RESISTANCE_SOURCES:
            raise ValueError(
                "PenmanMonteithET config must declare canopy_resistance_source as one of "
                f"{list(self.CANOPY_RESISTANCE_SOURCES)}, got {canopy_resistance_source!r}"
            )
        temporal_support = (config or {}).get("temporal_support", "hourly")
        if temporal_support not in self.TEMPORAL_SUPPORTS:
            raise ValueError(
                "PenmanMonteithET config temporal_support must be one of "
                f"{list(self.TEMPORAL_SUPPORTS)}, got {temporal_support!r}"
            )
        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Penman-Monteith evapotranspiration model with aerodynamic and surface resistance coupling",
            category=ProcessCategory.TRANSPORT_MECHANISM,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "ET = [Delta(Rn-G) + rho*cp*VPD*ga] / "
                "[lambda(Delta + gamma(1 + ga/gc))]; q_v = star_ET X_v"
            ),
            references=[
                "Monteith, J.L. (1965). Evaporation and environment. Symp. Soc. Exp. Biol. 19, 205-224",
                "Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation paper 56",
                "Penman, H.L. (1948). Natural evaporation from open water, bare soil and grass"
            ],
            parameters={
                "hydrosphere.evapotranspiration.soil_heat_flux_source": (
                    "Explicit ground heat source: fraction or signed state input [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.roughness_length": (
                    "Momentum roughness length z0m of the logarithmic wind profile [m]"
                ),
                "hydrosphere.evapotranspiration.displacement_height": (
                    "Zero-plane displacement height d of the logarithmic wind profile [m]"
                ),
                "hydrosphere.evapotranspiration.minimum_conductance": (
                    "Lower bound of the canopy conductance g_c [m s-1]"
                ),
                "hydrosphere.evapotranspiration.reference_height": (
                    "Measurement height z of wind speed and humidity [m]"
                ),
                "hydrosphere.evapotranspiration.enable_canopy_interception": (
                    "Switch of the capacity-overflow canopy interception store [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.interception_capacity_per_LAI": (
                    "Canopy water storage capacity per unit leaf area index [mm]"
                ),
                "hydrosphere.evapotranspiration.canopy_evaporation_efficiency": (
                    "Evaporation efficiency factor of the wet canopy [dimensionless]"
                ),
                "hydrosphere.evapotranspiration.hodge_reference_aerodynamic_conductance": (
                    "Reference aerodynamic conductance g_a of the static Hodge contract [m s-1]"
                ),
                "hydrosphere.physical.soil_heat_flux_ratio_daytime": (
                    "FAO-56 hourly soil heat flux ratio G/R_n for R_n > 0, Eq. 45 [dimensionless]"
                ),
                "hydrosphere.physical.soil_heat_flux_ratio_nighttime": (
                    "FAO-56 hourly soil heat flux ratio G/R_n for R_n <= 0, Eq. 46 [dimensionless]"
                ),
                "hydrosphere.physical.soil_heat_flux_hourly_max_step": (
                    "Longest step of the FAO-56 hourly soil heat flux formulation [s]"
                ),
                "hydrosphere.physical.reference_surface_resistance": (
                    "Surface resistance r_s of the FAO-56 reference grass [s m-1]"
                ),
                "hydrosphere.physical.heat_to_momentum_roughness_ratio": (
                    "Ratio z0h/z0m of the heat and momentum roughness lengths [dimensionless]"
                ),
                "hydrosphere.physical.atmospheric_pressure_min": (
                    "Lower plausibility bound of the atmospheric pressure input [Pa]"
                ),
                "hydrosphere.physical.atmospheric_pressure_max": (
                    "Upper plausibility bound of the atmospheric pressure input [Pa]"
                ),
                "hydrosphere.physical.maximum_lai_factor": (
                    "Upper leaf area index of the interception capacity checks [m2 m-2]"
                ),
                "hydrosphere.physical.moderate_lai_factor": (
                    "Representative leaf area index of the interception capacity scaling [m2 m-2]"
                ),
                "hydrosphere.numerical.temperature_bounds": (
                    "Plausibility bounds [min, max] of the air temperature input [K]"
                ),
                "hydrosphere.numerical.net_radiation_bounds": (
                    "Plausibility bounds [min, max] of the net radiation input [W m-2]"
                ),
                "hydrosphere.numerical.minimum_wind_speed_offset": (
                    "Offset of the wind speed plausibility check [m s-1]"
                ),
                "hydrosphere.numerical.aerodynamic_conductance_bounds": (
                    "Bounds [min, max] of the aerodynamic conductance g_a [m s-1]"
                ),
                "hydrosphere.numerical.evapotranspiration_flux_clamp_min": (
                    "Lower bound of the evapotranspiration flux [kg m-2 s-1]"
                ),
                "hydrosphere.numerical.minimum_wind_speed": (
                    "Lower bound of the wind speed in the aerodynamic conductance [m s-1]"
                ),
                "hydrosphere.numerical.canopy_storage_bounds": (
                    "Bounds [min, max] of the canopy water storage capacity [mm]"
                ),
                "hydrosphere.numerical.mass_conservation_tolerance_mm": (
                    "Absolute tolerance of the canopy water balance check [mm]"
                ),
                "hydrosphere.numerical.critical_mass_balance_error_threshold": (
                    "Critical error threshold of the canopy water balance [mm]"
                ),
                "hydrosphere.numerical.minimum_drainage_flux": (
                    "Precipitation-rate threshold of the interception efficiency diagnostic, "
                    "converted to a per-step depth by the liquid-water-equivalent rule [kg m-2 s-1]"
                ),
                "validation.numerical.clamp_min_zero": (
                    "Zero value of the non-negative flux clamps [dimensionless]"
                ),
            },
            input_variables=[
                "net_radiation",
                "soil_heat_flux",
                "air_temperature",
                "wind_speed",
                "atmospheric_pressure",
                "relative_humidity",
                "vapor_pressure_deficit",
                "aerodynamic_resistance",
                "precipitation_flux",
                "canopy_storage",
                "leaf_area_index",
            ],
            output_variables=[
                "evapotranspiration_flux",
                "latent_heat_flux",
                "diagnostics.vapor_pressure_deficit",
                "diagnostics.aerodynamic_conductance",
                "diagnostics.canopy_conductance",
                "canopy_storage",
                "diagnostics.evapotranspiration",
            ],
            # Penman-Monteith diagnoses an open atmosphere-surface water flux.
            # The latent-heat mapping is audited locally; reservoir closure
            # belongs to the coupled soil-canopy-atmosphere budget.
            conservation_laws=["latent_heat_boundary"],
            spatial_dimensions=0,  # Point-based model
            temporal_scale=TemporalScale.SECOND,
            coupling_interfaces=["soil_moisture", "energy_balance", "photosynthesis"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.FLUX,
            conflict_resolution_policy=ConflictResolutionPolicy.EXCLUSIVE,
        )
        daylight_rule = None
        if temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            daylight_rule = daylight_quadrature.daylight_config(config or {}, "PenmanMonteithET")
            humidity = ("relative_humidity", "vapor_pressure_deficit")
            metadata.input_variables[:] = [key for key in metadata.input_variables if key not in humidity]
            if canopy_resistance_source == "state":
                metadata.input_variables.append(self.DAYLIGHT_CONDUCTANCE_INPUT)
            metadata.input_variables.extend(
                key for key in self.DAYLIGHT_INPUTS if key not in metadata.input_variables)
        else:
            daylight_quadrature.reject_daylight_keys(config or {}, "PenmanMonteithET")
            if canopy_resistance_source == "state":
                metadata.input_variables.extend(self.STATE_RESISTANCE_INPUTS)
            if temporal_support == "daily":
                metadata.input_variables.extend(self.DAILY_TEMPERATURE_INPUTS)

        # Initialize parent classes (the label is the class attribute TRANSPORT_TYPE)
        super().__init__(name=name, config=config or {}, metadata=metadata)
        self.canopy_resistance_source = canopy_resistance_source
        self.temporal_support = temporal_support
        self._daylight_rule = daylight_rule
        # References: SI; relic.physical.constants.time.seconds_per_day.
        # Mathematical expression: dt_day = 86400 s, the one step of the daily support.
        # Units: dt_day [s].
        self._seconds_per_day = float(get_physical_constant("seconds_per_day"))
        self.soil_heat_flux_source = self.get_parameter_value(
            "soil_heat_flux_source", namespace="hydrosphere.evapotranspiration",
            source_order=("config", "manager"), expected_type=str, required=True,
        )
        if self.soil_heat_flux_source not in {"fraction", "state"}:
            raise ValueError("soil_heat_flux_source must be 'fraction' or 'state'")
        # Optional inputs and the physical meaning of their absence (never a numeric default):
        #   relative_humidity / vapor_pressure_deficit: one humidity state is required; RH is
        #     used when present, VPD only when RH is absent; both absent raises.
        #   surface_resistance / canopy_conductance (canopy_resistance_source "state" only): one
        #     canopy resistance r_s = 1 / g_c is required; r_s is used when present, 1 / g_c only
        #     when r_s is absent; both absent raises.
        #   aerodynamic_resistance: absent means g_a is computed from wind_speed (FAO-56 profile).
        #   precipitation_flux, canopy_storage, leaf_area_index: read only with canopy
        #     interception enabled, where each is required; without interception no canopy store
        #     is modelled, and a present canopy_storage / precipitation_flux is passed through.
        self.optional_input_variables = {
            "relative_humidity",
            "vapor_pressure_deficit",
            "aerodynamic_resistance",
            "precipitation_flux",
            "canopy_storage",
            "leaf_area_index",
        }
        if daylight_rule is not None:
            # daylight_gauss reads e_a (required) instead of either humidity state, and the
            #   required per-sample canopy conductance instead of r_s / g_c.
            self.optional_input_variables -= {"relative_humidity", "vapor_pressure_deficit"}
        elif canopy_resistance_source == "state":
            self.optional_input_variables.update(self.STATE_RESISTANCE_INPUTS)
        if self.soil_heat_flux_source == "fraction":
            self.optional_input_variables.add("soil_heat_flux")
        # The combination equation couples resistances, not a molecular diffusion law, so
        # the scalar diffusivity stays unset (None) and no vapour diffusivity is read.

        # Cache structural and numerical parameters
        self._load_diagnostic_defaults()
        
        # References: FAO-56 (Allen et al. 1998); Physical constants SSOT policy.
        # Mathematical expression: epsilon = M_water / M_air (psychrometric constant).
        # Parameter meanings: M_water water molecular weight, M_air dry air molecular weight.
        # Units: M_water, M_air [g mol-1]; epsilon [1].
        # Domain/constraints: SSOT constants; the moist-air density is not a constant and is
        #   computed per call by compute_moist_air_density from T, p and e_a.
        # Sign convention: not applicable (positive material constants).
        mw_water = float(get_physical_constant("water_molecular_weight"))
        mw_air = float(get_physical_constant("air_molecular_weight"))
        self.latent_heat_vaporization = float(get_physical_constant("latent_heat_vaporization"))
        self.specific_heat_air = float(get_physical_constant("air_specific_heat_cp"))
        self.molecular_weight_ratio = mw_water / mw_air
        self._tetens_base_pressure = float(
            get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_base_pressure"
            )
        )
        self._tetens_coeff_a = float(
            get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_coefficient_a"
            )
        )
        self._tetens_coeff_b = float(
            get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_coefficient_b"
            )
        )
        self._tetens_slope_coeff = float(
            get_physical_constant(
                "relic.physical.constants.psychrometrics.saturation_vapor_pressure_slope_coeff"
            )
        )
        self._celsius_to_kelvin = float(get_physical_constant("celsius_to_kelvin"))
        self.von_karman_constant = float(get_physical_constant("von_karman_constant"))
        
        canopy_params = {
            "roughness_length": "hydrosphere.evapotranspiration.roughness_length",
            "displacement_height": "hydrosphere.evapotranspiration.displacement_height",
            "minimum_conductance": "hydrosphere.evapotranspiration.minimum_conductance",
            "reference_height": "hydrosphere.evapotranspiration.reference_height",
            "enable_canopy_interception": "hydrosphere.evapotranspiration.enable_canopy_interception",
            "interception_capacity_per_LAI": "hydrosphere.evapotranspiration.interception_capacity_per_LAI",
            "canopy_evaporation_efficiency": "hydrosphere.evapotranspiration.canopy_evaporation_efficiency",
        }
        for attr, param_key in canopy_params.items():
            value = self.get_parameter_value(param_key, required=True)
            if attr == "enable_canopy_interception":
                value = bool(value)
            setattr(self, attr, value)

        # canopy_storage leaves this Functor only where a canopy store exists: with
        # interception it is the advanced store and is always returned; without interception
        # a present store is passed through unchanged and an absent one is neither read nor
        # invented, so the output is optional in that configuration only (the unit rule of
        # the validation layer skips an absent optional output).
        if not self.enable_canopy_interception:
            self.metadata.optional_output_variables = list(
                dict.fromkeys([*self.metadata.optional_output_variables, "canopy_storage"])
            )

        # Note: Canopy interception state is managed by StateManager; this functor remains stateless.

    @staticmethod
    def _scalar_from_value(value: Union[float, int, Tensor]) -> float:
        """Convert numeric values to Python float without tensor.item/cpu APIs."""
        if isinstance(value, Tensor):
            detached = value.detach().reshape(-1)
            if detached.numel() == 0:
                raise ValueError("Cannot convert empty tensor to scalar.")
            return detached[:1].to(dtype=torch.float64).tolist()[0]
        return float(value)

    def _load_diagnostic_defaults(self) -> None:
        """Load the structural reference conductance and numerical constants."""

        def to_float(value: Union[float, Tensor]) -> float:
            return self._scalar_from_value(value)

        diag_namespace = "hydrosphere.evapotranspiration"
        # Reference aerodynamic conductance of the static Hodge contract
        # (build_hodge_contract); never substituted for a runtime state.
        self._hodge_reference_aerodynamic_conductance = self._ensure_tensor(
            self._require_manager_param(
                diag_namespace,
                "hodge_reference_aerodynamic_conductance",
            )
        )

        self._zero_flux = self._ensure_tensor(
            self._require_manager_param(
                "validation.numerical",
                "clamp_min_zero",
            )
        )

        self._seconds_per_hour = to_float(get_physical_constant("seconds_per_hour"))
    
    def get_dependencies(self) -> DependencyDeclaration:
        """
        Return list of dependencies for this functor.
        
        Returns
        -------
        DependencyDeclaration
            Lagged dependencies on soil moisture providers
        """
        return self._merge_declared_dependencies({
            "lagged": ["bucket_soil_moisture", "richards_soil_moisture"]
        })

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
        """Declare canonical cochain spaces for Penman-Monteith ET.

        References:
          - Penman (1948), Monteith (1965), Allen et al. (1998), Hirani (2003).
        Mathematical expression:
          q_v = star_ET X_v, lambda q_v = L_E, with X_v = VPD.
        Parameter meanings:
          q_v is water vapor mass flux [kg m-2 s-1], L_E is latent heat flux
          [W m-2], and VPD is vapor pressure deficit [Pa].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_penman_monteith_et",
                degree=1,
                primality="dual",
                n_dofs=n_dofs,
                units="kg m-2 s-1",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="evapotranspiration_flux",
                orientation_convention="outward_positive",
                boundary_role="port",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_penman_monteith_et",
                degree=1,
                primality="dual",
                n_dofs=n_dofs,
                units="W m-2",
                extensivity="integral",
                physical_domain="hydrosphere",
                semantic_name="latent_heat_flux",
                orientation_convention="outward_positive",
                boundary_role="port",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_penman_monteith_et",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="Pa",
                extensivity="intensive",
                physical_domain="atmosphere",
                semantic_name="vapor_pressure_deficit",
                orientation_convention="surface_to_atmosphere_positive",
                boundary_role="interior",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "hydrosphere.evapotranspiration.soil_heat_flux_source",
            "hydrosphere.evapotranspiration.roughness_length",
            "hydrosphere.evapotranspiration.displacement_height",
            "hydrosphere.evapotranspiration.minimum_conductance",
            "hydrosphere.evapotranspiration.reference_height",
            "hydrosphere.evapotranspiration.enable_canopy_interception",
            "hydrosphere.evapotranspiration.interception_capacity_per_LAI",
            "hydrosphere.evapotranspiration.canopy_evaporation_efficiency",
            "hydrosphere.evapotranspiration.hodge_reference_aerodynamic_conductance",
            "hydrosphere.physical.soil_heat_flux_ratio_daytime",
            "hydrosphere.physical.soil_heat_flux_ratio_nighttime",
            "hydrosphere.physical.soil_heat_flux_hourly_max_step",
            "hydrosphere.physical.reference_surface_resistance",
            "hydrosphere.physical.heat_to_momentum_roughness_ratio",
            "hydrosphere.physical.atmospheric_pressure_min",
            "hydrosphere.physical.atmospheric_pressure_max",
            "hydrosphere.physical.maximum_lai_factor",
            "hydrosphere.physical.moderate_lai_factor",
            "hydrosphere.numerical.temperature_bounds",
            "hydrosphere.numerical.net_radiation_bounds",
            "hydrosphere.numerical.minimum_wind_speed_offset",
            "hydrosphere.numerical.aerodynamic_conductance_bounds",
            "hydrosphere.numerical.evapotranspiration_flux_clamp_min",
            "hydrosphere.numerical.minimum_wind_speed",
            "hydrosphere.numerical.canopy_storage_bounds",
            "hydrosphere.numerical.mass_conservation_tolerance_mm",
            "hydrosphere.numerical.critical_mass_balance_error_threshold",
            "hydrosphere.numerical.minimum_drainage_flux",
            "validation.numerical.clamp_min_zero",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the vapor transport Hodge contract for Penman-Monteith ET.

        References:
          - Monteith (1965), Allen et al. (1998), and van der Schaft and Maschke (1995).
        Mathematical expression:
          star_ET = diag(g_eff / lambda), g_eff = (g_a g_c) / (g_a + g_c).
        Parameter meanings:
          g_a is aerodynamic conductance [m s-1], g_c is canopy conductance
          [m s-1], and lambda is latent heat of vaporization [J kg-1].
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        # References: Monteith (1965); Allen et al. (1998), FAO-56; van der Schaft and
        #   Maschke (1995).
        # Mathematical expression: g_eff = g_a g_c / (g_a + g_c); m = g_eff / lambda.
        # Parameter meanings: g_a the reference aerodynamic conductance
        #   (hydrosphere.evapotranspiration.hodge_reference_aerodynamic_conductance); g_c the
        #   minimum canopy conductance (hydrosphere.evapotranspiration.minimum_conductance); lambda
        #   the latent heat of vaporization (physical-constant SSOT); m the Hodge mobility.
        # Units: g_a, g_c, g_eff [m s-1]; lambda [J kg-1]; m [s m-1] (= [m s-1] / [J kg-1]),
        #   so m times the vapour pressure deficit effort [Pa] is a mass flux [kg m-2 s-1].
        # Domain/constraints: g_a > 0, g_c > 0 and lambda > 0 by their parameter bounds and
        #   the constant SSOT, hence g_a + g_c > 0 and m > 0. A non-finite or non-positive
        #   value raises; no dimensionless floor lifts a conductance or lambda.
        # Sign convention: m is a positive mobility; the flux orientation is the declared
        #   codomain orientation (outward_positive).
        g_a = self._scalar_from_value(self._hodge_reference_aerodynamic_conductance)
        g_c = self._scalar_from_value(self.minimum_conductance)
        latent_heat = float(self.latent_heat_vaporization)
        if not all(math.isfinite(value) and value > 0.0 for value in (g_a, g_c, latent_heat)):
            raise ValueError(
                "Penman-Monteith Hodge contract requires positive finite g_a, g_c and lambda; "
                f"got g_a={g_a} m s-1, g_c={g_c} m s-1, lambda={latent_heat} J kg-1"
            )
        effective_conductance = (g_a * g_c) / (g_a + g_c)
        mobility_value = effective_conductance / latent_heat
        values = torch.full(
            (n_dofs,),
            mobility_value,
            dtype=torch.float32,
            device=getattr(self, "device", torch.device("cpu")),
        )
        if bool(torch.any(values <= 0.0)):
            raise ValueError("Penman-Monteith vapor transport mobility must be positive.")

        domain = CochainSpaceSpec(
            complex_id="hydrosphere_penman_monteith_et",
            degree=0,
            primality="primal",
            n_dofs=n_dofs,
            units="Pa",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="vapor_pressure_deficit_effort",
            orientation_convention="surface_to_atmosphere_positive",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_penman_monteith_et",
            degree=1,
            primality="dual",
            n_dofs=n_dofs,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="evapotranspiration_mass_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_penman_monteith_vapor_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=(
                "hydrosphere.evapotranspiration.hodge_reference_aerodynamic_conductance",
                "hydrosphere.evapotranspiration.minimum_conductance",
            ),
            metric_keys=("hydrosphere.evapotranspiration.reference_height",),
            material_law="penman_monteith_vapor_transport_mobility",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build Penman-Monteith Hodge terms for IPHS assembly."""

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
            "vapor_transport_dissipation": "sigma = X_v.T star_ET X_v / T >= 0",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for scalar ET transport."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the positive vapor transport mobility contribution.

        References:
          - Monteith (1965), Allen et al. (1998), and van der Schaft and Maschke (1995).
        Mathematical expression:
          M_h = star_ET, sigma = X_v.T M_h X_v >= 0.
        Parameter meanings:
          X_v is vapor pressure deficit [Pa].
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"vapor_pressure_deficit_effort[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare Penman-Monteith vapor and latent heat ports."""

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_penman_monteith_et",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="Pa",
            extensivity="intensive",
            physical_domain="atmosphere",
            semantic_name="vapor_pressure_deficit_effort",
            orientation_convention="surface_to_atmosphere_positive",
            boundary_role="port",
        )
        vapor_flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_penman_monteith_et",
            degree=1,
            primality="dual",
            n_dofs=1,
            units="kg m-2 s-1",
            extensivity="integral",
            physical_domain="hydrosphere",
            semantic_name="penman_monteith_vapor_mass_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        heat_flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_penman_monteith_et",
            degree=1,
            primality="dual",
            n_dofs=1,
            units="W m-2",
            extensivity="integral",
            physical_domain="surface_energy_balance",
            semantic_name="penman_monteith_latent_heat_flux",
            orientation_convention="outward_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="penman_monteith_vapor_flux",
                effort_var="vapor_pressure_deficit",
                flow_var="evapotranspiration_flux",
                effort_space=effort_space,
                flow_space=vapor_flow_space,
                power_units="Pa kg m-2 s-1",
                positive_power_convention="out_of_system",
                port_kind="matter",
            ),
            PortSpecIPHS(
                name="penman_monteith_latent_heat_flux",
                effort_var="surface_energy_gradient",
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
        #   Monteith (1965); Allen et al. (1998), FAO-56.
        # Mathematical expression:
        #   ET_raw = [Delta*(Rn-G)+rho_a*cp*VPD*ga] / [Lv*(Delta+gamma*(1+ga/gc))];
        #   ET = max(ET_raw, ET_min); LE = Lv*ET.
        # Parameter meanings:
        #   Delta is the saturation-pressure slope; gamma is the psychrometric constant;
        #   Rn-G is available energy; ga=1/ra and gc=1/rs are conductances.
        # Units:
        #   ET [kg m-2 s-1], LE/Rn/G [W m-2], Lv [J kg-1],
        #   Delta/gamma [Pa K-1], VPD [Pa], ga/gc [m s-1].
        # Domain/constraints:
        #   Finite declared meteorology and positive resistances; use the supplied ra
        #   when present. Outputs are REPLACE rates; negative-ET projection is diagnosed.
        #   Canopy interception is a separate selected substep whose storage budget is
        #   validated on one time level: S_old + P = S_new + E_i + T_fall.
        # Sign convention:
        #   Positive ET and LE export water and latent heat from the surface.
                state: StateDict,
                time: float,
                dt: float,
                **kwargs) -> FluxDict:
        """
        Compute evapotranspiration using Penman-Monteith equation.

        Parameters
        ----------
        state : StateDict
            Current system state containing environmental variables
        time : float
            Current simulation time [s]
        dt : float
            Time step [s]
        **kwargs
            Not read for inputs: the state is the one input channel (declared inputs).

        Returns
        -------
        FluxDict
            Evapotranspiration flux and diagnostic variables

        Raises
        ------
        ValueError
            If required state variables are missing or invalid
        """
        # The state is the one input channel: no environment keyword supplies inputs.
        combined_state = dict(state)

        # The rate law (_penman_monteith_law) reads the declared inputs: g_a = 1 / r_a from the
        #   optional aerodynamic_resistance input, or, without it, the FAO-56 Eq. 4 neutral
        #   log-profile conductance from wind_speed (a documented optional branch).
        if self.temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            law = self._daylight_gauss_law(combined_state, dt)
        else:
            law = self._penman_monteith_law(combined_state, dt, self.temporal_support)
        net_radiation = law["net_radiation"]  # W m-2
        soil_heat_flux = law["soil_heat_flux"]  # W m-2
        available_energy = law["available_energy"]  # W m-2
        vapor_pressure_deficit = law["vapor_pressure_deficit"]  # Pa
        aerodynamic_conductance = law["aerodynamic_conductance"]  # m s-1
        canopy_conductance = law["canopy_conductance"]  # m s-1
        slope_svp = law["slope_svp"]  # Pa K-1
        psychrometric_constant = law["psychrometric_constant"]  # Pa K-1
        evapotranspiration_flux = law["evapotranspiration_flux"]  # kg m-2 s-1
        negative_et_projection_residual = law["negative_et_projection_residual"]  # kg m-2 s-1
        reference_et = law["reference_et"]  # kg m-2 s-1
        equilibrium_et = law["equilibrium_et"]  # kg m-2 s-1
        zero_floor = self._ensure_tensor(self._zero_flux)

        # References: Rutter et al. (1971); Gash (1979); RHESSys canopy_et.c.
        # Mathematical expression: C_max = c_LAI * LAI; I = min(P, max(0, C_max - S_old));
        #   E_i = min(S_old + I, eta * ET * dt); Drip = max(0, S_old + I - E_i - C_max);
        #   S_new = S_old + I - E_i - Drip; T_fall = (P - I) + Drip; T = ET*dt - E_i.
        # Parameter meanings: c_LAI interception_capacity_per_LAI, eta
        #   canopy_evaporation_efficiency, LAI leaf area index, P precipitation
        #   depth, ET Penman-Monteith demand, S canopy storage, T transpiration.
        # Units: c_LAI [mm], eta [1], LAI [m2 m-2], P,I,E_i,Drip,S,T_fall,T [mm];
        #   precipitation_flux and ET [kg m-2 s-1] with 1 kg m-2 = 1 mm; dt [s].
        # Domain/constraints: nonnegative depths; C_max clamped to
        #   hydrosphere.numerical.canopy_storage_bounds; the checker enforces the
        #   complete budget S_old + P = S_new + E_i + T_fall at consistent time levels.
        # Sign convention: storage change S_new - S_old is negative when the
        #   canopy drains; all fluxes are positive magnitudes leaving or entering the store.
        # 7. Canopy Interception Processing (P2 Enhancement)
        if self.enable_canopy_interception:
            (
                evapotranspiration_flux,
                evap_from_interception,
                transpiration_flux,
                throughfall_flux,
                storage_change,
                new_canopy_storage  # NEW: Receive updated storage value
            ) = self._process_canopy_interception(
                evapotranspiration_flux, combined_state, dt
            )
        else:
            # Standard mode: all ET is transpiration and no canopy store is modelled; a
            # present precipitation_flux passes through as throughfall and a present
            # canopy_storage is returned unchanged; absent ones are neither read nor invented.
            evap_from_interception = self._zero_flux.clone()
            transpiration_flux = evapotranspiration_flux
            throughfall_flux = (
                self._ensure_tensor(combined_state["precipitation_flux"])
                if combined_state.get("precipitation_flux") is not None
                else None
            )
            storage_change = self._zero_flux.clone()
            new_canopy_storage = (
                self._ensure_tensor(combined_state["canopy_storage"])
                if combined_state.get("canopy_storage") is not None
                else None
            )
        available_energy_residual = net_radiation - soil_heat_flux - available_energy
        canopy_storage_residual = (
            torch.clamp(-new_canopy_storage, min=zero_floor)
            if new_canopy_storage is not None
            else None
        )
        
        diagnostics = {
            "soil_heat_flux": soil_heat_flux,                    # W m-2, positive into soil
            "available_energy": available_energy,              # W m-2, signed Rn-G
            "reference_evapotranspiration": reference_et,        # kg m-2 s-1
            "equilibrium_evapotranspiration": equilibrium_et,    # kg m-2 s-1
            "vapor_pressure_deficit": vapor_pressure_deficit,    # Pa
            "aerodynamic_conductance": aerodynamic_conductance,  # m s-1
            "canopy_conductance": canopy_conductance,            # m s-1
            "slope_saturation_vapor_pressure": slope_svp,        # Pa K-1
            "psychrometric_constant": psychrometric_constant,    # Pa K-1
            "evaporation_from_interception": evap_from_interception,  # kg m-2 s-1
            "transpiration_flux": transpiration_flux,                 # kg m-2 s-1
            "throughfall_flux": throughfall_flux,                     # kg m-2 s-1
            "canopy_storage_change": storage_change,                  # mm per step, S_new - S_old
            "canopy_interception_storage": new_canopy_storage,         # mm
            "available_energy_residual": available_energy_residual,    # W m-2
            "canopy_storage_residual": canopy_storage_residual,        # mm
            "negative_et_projection_residual": negative_et_projection_residual,  # kg m-2 s-1
        }
        # Without canopy interception and without the optional inputs, there is no
        # throughfall pass-through and no canopy store to report.
        diagnostics = {key: value for key, value in diagnostics.items() if value is not None}

        outputs = {
            "evapotranspiration_flux": UpdateItem(
                value=evapotranspiration_flux,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="kg m-2 s-1",
                source_functor=self.name,
            ),
            "latent_heat_flux": UpdateItem(
                value=evapotranspiration_flux * self.latent_heat_vaporization,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="W m-2",
                source_functor=self.name,
            ),
            "diagnostics.vapor_pressure_deficit": UpdateItem(
                value=vapor_pressure_deficit,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="Pa",
                source_functor=self.name,
            ),
            "diagnostics.aerodynamic_conductance": UpdateItem(
                value=aerodynamic_conductance,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="m s-1",
                source_functor=self.name,
            ),
            "diagnostics.canopy_conductance": UpdateItem(
                value=canopy_conductance,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="m s-1",
                source_functor=self.name,
            ),
            "diagnostics.evapotranspiration": diagnostics,
        }
        if new_canopy_storage is not None:
            outputs["canopy_storage"] = UpdateItem(
                value=new_canopy_storage,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit="mm",
                source_functor=self.name,
            )
        return outputs

    def _penman_monteith_law(self, combined_state: Dict[str, Any], dt: float, support: str) -> Dict[str, Tensor]:
        """Penman-Monteith rate law on one temporal support (no canopy store)."""
        # References: Monteith (1965); Allen et al. (1998), FAO-56 (see forward).
        # Mathematical expression: the combination equation of forward; support selects
        #   e_s ("daily": FAO-56 Eq. 12; otherwise e0(T) of the given air temperature),
        #   the step-domain check ("hourly": dt <= dt_max; "daily": dt = dt_day;
        #   "daylight_sample": none here, the daylight caller checks dt = dt_day) and G
        #   ("daily": Eq. 42; "hourly" and "daylight_sample": Eqs. 45-46 fractions).
        # Parameter meanings: support one of "hourly", "daily", "daylight_sample".
        # Units: as forward; the returned rates are [kg m-2 s-1], energies [W m-2].
        # Domain/constraints: the input checks below apply elementwise (every sample).
        # Sign convention: as forward.
        # Extract required state variables with validation
        try:
            net_radiation = self._ensure_tensor(combined_state["net_radiation"])  # W m-2
            air_temp = self._ensure_tensor(combined_state["air_temperature"])     # K
            wind_speed = self._ensure_tensor(combined_state["wind_speed"])        # m s-1
        except KeyError as e:
            raise ValueError(f"Required state variable missing for Penman-Monteith ET: {e}")

        # Atmospheric pressure is required for psychrometric constant calculation.
        if "atmospheric_pressure" in combined_state:
            atm_pressure_raw = combined_state["atmospheric_pressure"]
        else:
            atm_pressure_raw = None
        if atm_pressure_raw is None:
            raise ValueError("Required state variable missing for Penman-Monteith ET: 'atmospheric_pressure'")
        atm_pressure = self._ensure_tensor(atm_pressure_raw)  # Pa

        if "relative_humidity" in combined_state:
            relative_humidity_raw = combined_state["relative_humidity"]
        else:
            relative_humidity_raw = None
        if "vapor_pressure_deficit" in combined_state:
            vapor_pressure_deficit_raw = combined_state["vapor_pressure_deficit"]
        else:
            vapor_pressure_deficit_raw = None

        # Canopy (surface) resistance from the declared source:
        # References: Monteith (1965); Allen et al. (1998, FAO-56), reference grass r_s.
        # Mathematical expression: r_s = r_s,state, r_s = 1 / g_c, or r_s = r_s,ref (FAO-56).
        # Parameter meanings: r_s surface (canopy) resistance; g_c canopy conductance;
        #   r_s,ref hydrosphere.physical.reference_surface_resistance.
        # Units: r_s [s m-1]; g_c [m s-1].
        # Domain/constraints: r_s > 0; g_c clamped below by minimum_conductance.
        # Sign convention: resistances and conductances are positive magnitudes.
        if self.canopy_resistance_source == "fao56_reference_surface":
            surface_resistance = self._ensure_tensor(
                self._require_manager_param(
                    "hydrosphere.physical",
                    "reference_surface_resistance",
                )
            )  # s m-1
        elif "surface_resistance" in combined_state and combined_state["surface_resistance"] is not None:
            surface_resistance = self._ensure_tensor(combined_state["surface_resistance"])  # s m-1
        elif "canopy_conductance" in combined_state and combined_state["canopy_conductance"] is not None:
            g_c = self._ensure_tensor(combined_state["canopy_conductance"])  # m s-1
            g_c = torch.clamp(g_c, min=self.minimum_conductance)
            surface_resistance = 1.0 / g_c
        else:
            raise ValueError(
                "Required canopy resistance missing for Penman-Monteith ET: provide "
                "'surface_resistance' [s m-1] or 'canopy_conductance' [m s-1] "
                "(a leaf-level stomatal_conductance is not a canopy conductance)."
            )
        
        # References: WMO Guide to Meteorological Instruments; FAO-56.
        # Mathematical expression: enforce physical bounds for T, Rn, and P.
        # Parameter meanings: temperature_bounds [K], net_radiation_bounds [W m-2],
        # atmospheric_pressure_min/max [Pa].
        temp_bounds = self.get_parameter_value(
            "temperature_bounds",
            namespace="hydrosphere.numerical", required=True)
        temp_min = float(temp_bounds[0])
        temp_max = float(temp_bounds[1])
        validate_temperature_range(air_temp, min_temp=temp_min, max_temp=temp_max)
        net_radiation_bounds = self.get_parameter_value(
            "net_radiation_bounds",
            namespace="hydrosphere.numerical", required=True)
        net_radiation_min = float(net_radiation_bounds[0])
        net_radiation_max = float(net_radiation_bounds[1])
        if torch.any(net_radiation < net_radiation_min) or torch.any(net_radiation > net_radiation_max):
            raise ValueError(
                f"Net radiation outside physical bounds [{net_radiation_min}, {net_radiation_max}] W/m2"
            )
        wind_speed_offset = self._require_manager_param(
            "hydrosphere.numerical",
            "minimum_wind_speed_offset",
        )
        validate_positive_values(wind_speed + wind_speed_offset, "wind_speed (with offset)")  # Allow very low wind
        validate_positive_values(surface_resistance, "surface_resistance")
        pressure_min = self.get_parameter_value(
            "atmospheric_pressure_min",
            namespace="hydrosphere.physical", required=True)
        pressure_max = self.get_parameter_value(
            "atmospheric_pressure_max",
            namespace="hydrosphere.physical", required=True)
        if torch.any(atm_pressure < pressure_min) or torch.any(atm_pressure > pressure_max):
            raise ValueError(
                f"Atmospheric pressure outside reasonable range [{pressure_min}, {pressure_max}] Pa"
            )
        
        # 1. Compute psychrometric variables
        # Saturation vapor pressure using corrected Tetens equation (FAO-56)
        saturation_vapor_pressure = self._es_tetens_pa(air_temp)  # Pa
        # On the daily support e_s is the FAO-56 Eq. 12 mean of e_s(T_max) and e_s(T_min).
        if support == "daily":
            saturation_vapor_pressure = self._daily_saturation_vapor_pressure(combined_state)  # Pa

        # Humidity/VPD input continuity:
        # References: Allen et al. (1998, FAO-56); Monteith (1965)
        # Mathematical expressions:
        #   VPD = e_s(T) - e_a
        #   RH = e_a / e_s(T) = 1 - VPD / e_s(T)
        # Parameter meanings:
        #   RH: relative humidity [0-1]
        #   VPD: vapor pressure deficit [Pa]
        #   e_s(T): saturation vapor pressure at air temperature [Pa]
        zero_floor = self._ensure_tensor(self._zero_flux)
        if relative_humidity_raw is not None:
            relative_humidity = self._ensure_tensor(relative_humidity_raw)  # 0-1
            if torch.any(relative_humidity < 0) or torch.any(relative_humidity > 1):
                raise ValueError("Relative humidity must be in range [0, 1]")
            actual_vapor_pressure = saturation_vapor_pressure * relative_humidity
            vapor_pressure_deficit = torch.clamp(
                saturation_vapor_pressure - actual_vapor_pressure,
                min=zero_floor,
            )  # Pa
        elif vapor_pressure_deficit_raw is not None:
            vapor_pressure_deficit = self._ensure_tensor(vapor_pressure_deficit_raw)  # Pa
            vapor_pressure_deficit = torch.clamp(vapor_pressure_deficit, min=zero_floor)
            actual_vapor_pressure = torch.clamp(
                saturation_vapor_pressure - vapor_pressure_deficit,
                min=zero_floor,
            )
            relative_humidity = torch.clamp(
                actual_vapor_pressure / saturation_vapor_pressure,
                min=zero_floor,
                max=1.0,
            )
        else:
            raise ValueError(
                "Required humidity state missing for Penman-Monteith ET: provide "
                "'relative_humidity' [0-1] or 'vapor_pressure_deficit' [Pa]."
            )
        
        # Slope of saturation vapor pressure curve (corrected)
        slope_svp = self._slope_svp_pa_per_K(air_temp)  # Pa K-1
        
        # Psychrometric constant
        psychrometric_constant = compute_psychrometric_constant(
            atm_pressure,
            specific_heat_air=self.specific_heat_air,
            molecular_weight_ratio=self.molecular_weight_ratio,
            latent_heat=self.latent_heat_vaporization,
        )  # Pa K-1
        
        # 2. Compute aerodynamic conductance
        # References: Allen et al. (1998), FAO-56 Eq. 4; Monteith and Unsworth (2013).
        # Mathematical expression: g_a = 1 / r_a with r_a the optional
        #   aerodynamic_resistance input; without it, the FAO-56 Eq. 4 neutral profile
        #   g_a = k^2 u_z / (ln((z - d)/z_om) ln((z - d)/z_oh)); g_a is bounded by
        #   aerodynamic_conductance_bounds in both branches.
        # Parameter meanings: r_a the aerodynamic resistance of the turbulence owner, u_z
        #   wind speed at the reference height z, d displacement height, z_om and z_oh
        #   momentum and heat roughness lengths, k the von Karman constant.
        # Units: r_a [s m-1]; g_a and u_z [m s-1]; z, d, z_om, z_oh [m]; k [1].
        # Domain/constraints: aerodynamic_resistance is a declared optional input with the
        #   literature-defined FAO-56 computation as its absent-input meaning (an explicit
        #   branch, not a substitute value); r_a > 0 when supplied.
        # Sign convention: g_a is a positive transfer magnitude.
        if "aerodynamic_resistance" in combined_state:
            # Convert resistance to conductance (1/resistance)
            aerodynamic_resistance = self._ensure_tensor(combined_state["aerodynamic_resistance"])  # s m-1
            validate_positive_values(aerodynamic_resistance, "aerodynamic_resistance")
            ga_bounds = self._require_manager_param(
                "hydrosphere.numerical",
                "aerodynamic_conductance_bounds",
            )
            aerodynamic_conductance = torch.clamp(
                1.0 / aerodynamic_resistance,
                min=ga_bounds[0], max=ga_bounds[1]
            )  # m s-1
        else:
            # FAO-56 Eq. 4 neutral log-profile conductance from wind_speed.
            aerodynamic_conductance = self._compute_aerodynamic_conductance(
                wind_speed, air_temp
            )  # m s-1
        
        # 3. Convert surface resistance to canopy conductance
        # Surface resistance is resistance, conductance is 1/resistance
        canopy_conductance = torch.clamp(
            1.0 / surface_resistance, 
            min=self.minimum_conductance
        )  # m s-1
        
        # References: Monteith (1965); Allen et al. (1998), FAO-56 Eqs. 3, 42, 45-46.
        # Mathematical expression: G=G_state for source state; otherwise, hourly,
        #   G=c_day*Rn if Rn>0, else G=c_night*Rn (Eqs. 45-46), and, daily, G=0
        #   (Eq. 42: the day-scale soil heat flux beneath a grass reference surface is
        #   relatively small and may be ignored); A=Rn-G. Both sources feed one PM law.
        # Parameter meanings: G_state measured or coupled surface soil heat;
        #   c_day/c_night declared radiation fractions; Rn net radiation;
        #   A available energy; dt_max soil_heat_flux_hourly_max_step; dt_day the
        #   seconds_per_day constant.
        # Units: Rn/G/G_state/A [W m-2]; c_day/c_night [1]; dt/dt_max/dt_day [s].
        # Domain/constraints: finite signed G on the same area/time support as Rn;
        #   missing G never selects a different source. The hourly support keeps its
        #   step domain dt <= dt_max for both sources; the daily support is qualified
        #   for dt = dt_day only (Eq. 42 is a 24-hour statement), for both sources.
        # Sign convention: Rn enters the surface; G enters soil and may be negative;
        #   negative A is retained for the subsequent explicit ET projection.
        hourly_max_step = float(
            self._require_manager_param("hydrosphere.physical", "soil_heat_flux_hourly_max_step")
        )  # s
        if support == "hourly" and float(dt) > hourly_max_step:
            raise ValueError(
                f"PenmanMonteithET step dt={float(dt)} s exceeds {hourly_max_step} s, the validity "
                "limit of the FAO-56 hourly soil heat flux fractions (Eqs. 45-46)"
            )
        if support == "daily" and float(dt) != self._seconds_per_day:
            raise ValueError(
                f"PenmanMonteithET daily support requires dt={self._seconds_per_day} s, "
                f"got dt={float(dt)} s (FAO-56 Eqs. 12 and 42 are 24-hour forms)"
            )
        if self.soil_heat_flux_source == "state":
            if "soil_heat_flux" not in combined_state:
                raise ValueError("Required state variable soil_heat_flux is missing")
            soil_heat_flux = self._ensure_tensor(combined_state["soil_heat_flux"])
            if not bool(torch.isfinite(soil_heat_flux).all()):
                raise ValueError("soil_heat_flux must contain only finite values")
        elif support == "daily":
            soil_heat_flux = torch.zeros_like(net_radiation)  # W m-2, FAO-56 Eq. 42
        else:
            soil_heat_flux_ratio = torch.where(
                net_radiation > 0,
                self._ensure_tensor(
                    self._require_manager_param("hydrosphere.physical", "soil_heat_flux_ratio_daytime")
                ),
                self._ensure_tensor(
                    self._require_manager_param("hydrosphere.physical", "soil_heat_flux_ratio_nighttime")
                ),
            )
            soil_heat_flux = soil_heat_flux_ratio * net_radiation  # W m-2
        available_energy = net_radiation - soil_heat_flux  # W m-2
        
        # References: Monteith (1965); Allen et al. (1998); RHESSys penman_monteith.c.
        # Equation: ET = (Delta * Rn + rho * cp * VPD * ga) / (lambda * (Delta + gamma * (1 + ga / gc))).
        # Parameters: Delta slope of saturation vapor pressure curve [Pa K-1],
        # Rn net radiation minus soil heat flux [W m-2], rho air density [kg m-3],
        # cp specific heat of air [J kg-1 K-1], VPD vapor pressure deficit [Pa],
        # ga aerodynamic conductance [m s-1], gc canopy conductance [m s-1],
        # lambda latent heat of vaporization [J kg-1], gamma psychrometric constant [Pa K-1].
        # 5. Apply Penman-Monteith equation
        # Moist-air density of the aerodynamic term (actual and reference ET):
        # References: Holton and Hakim (2012) Eqs. 2.63-2.64; Allen et al. (1998), FAO-56 Annex 3.
        # Mathematical expression: rho_a = (p - (1 - epsilon) e_a) / (R_d T).
        # Parameter meanings: p atmospheric_pressure; e_a actual vapour pressure of the RH or
        #   VPD path above; T air_temperature; epsilon = M_w / M_d; R_d = R / M_d.
        # Units: rho_a [kg m-3]; p, e_a [Pa]; T [K].
        # Domain/constraints: T > 0, p > 0, 0 <= e_a < p (compute_moist_air_density raises).
        # Sign convention: rho_a is a positive state property.
        air_density = compute_moist_air_density(
            air_temp, atm_pressure, actual_vapor_pressure
        )  # kg m-3
        # Numerator: energy term + aerodynamic term
        energy_term = slope_svp * available_energy
        aerodynamic_term = (
            air_density * self.specific_heat_air *
            vapor_pressure_deficit * aerodynamic_conductance
        )
        numerator = energy_term + aerodynamic_term
        
        # Denominator: combined slope and psychrometric terms.
        # References: Monteith (1965); Allen et al. (1998), FAO-56, Eqs. 3, 8 and 13.
        # Mathematical expression: D = lambda (Delta + gamma (1 + g_a / g_c)) with
        #   Delta = c e0 exp(a Tc / (Tc + b)) / (Tc + b)^2, gamma = c_p p / (epsilon lambda)
        #   and g_c = max(1 / r_s, g_min); hence D >= lambda (Delta + gamma) > 0.
        # Parameter meanings: lambda latent heat of vaporization; Delta slope of the
        #   saturation vapour pressure curve (Tetens constants e0, a, b, c); gamma
        #   psychrometric constant; c_p specific heat of air; epsilon = M_w / M_d; p
        #   atmospheric pressure; g_a aerodynamic conductance; g_c canopy conductance;
        #   g_min hydrosphere.evapotranspiration.minimum_conductance.
        # Units: D [Pa K-1 J kg-1]; lambda [J kg-1]; Delta, gamma [Pa K-1]; c_p
        #   [J kg-1 K-1]; p [Pa]; Tc, b [degC]; g_a, g_c, g_min [m s-1].
        # Domain/constraints: the checks above hold T inside
        #   hydrosphere.numerical.temperature_bounds (Tc + b > 0, so Delta > 0) and p inside
        #   [atmospheric_pressure_min, atmospheric_pressure_max] (p > 0, so gamma > 0);
        #   g_a lies in hydrosphere.numerical.aerodynamic_conductance_bounds (lower bound > 0)
        #   and g_c >= g_min > 0. D therefore stays positive; a dimensionless epsilon cannot
        #   bound a quantity in Pa K-1 J kg-1, so a non-finite or non-positive D is a
        #   violated input contract and raises instead of being clamped.
        # Sign convention: D is a positive magnitude; ET takes the sign of the numerator
        #   (positive ET exports water from the surface).
        resistance_factor = aerodynamic_conductance / torch.clamp(
            canopy_conductance, min=self.minimum_conductance
        )
        denominator = self.latent_heat_vaporization * (
            slope_svp + psychrometric_constant * (1.0 + resistance_factor)
        )
        if not bool(torch.all(torch.isfinite(denominator) & (denominator > 0.0))):
            raise ValueError(
                "Penman-Monteith denominator lambda (Delta + gamma (1 + g_a / g_c)) must be "
                "finite and positive on the validated domain; check air_temperature, "
                "atmospheric_pressure and the conductances"
            )

        # Calculate evapotranspiration flux [kg m-2 s-1]
        raw_evapotranspiration_flux = numerator / denominator
        negative_et_projection_residual = torch.clamp(
            -raw_evapotranspiration_flux,
            min=zero_floor,
        )
        
        # Ensure non-negative ET (physical constraint)
        evapotranspiration_flux = torch.clamp(
            raw_evapotranspiration_flux,
            min=self._require_manager_param(
                "hydrosphere.numerical",
                "evapotranspiration_flux_clamp_min",
            ),
        )
        
        # References: Allen et al. (1998) FAO-56; Monteith and Unsworth (1990).
        # Equation: ET_ref = (Delta * Rn + rho * cp * VPD * ga) /
        #           (lambda * (Delta + gamma * (1 + ga / g_ref))).
        # Parameters: g_ref reference canopy conductance [m s-1], other parameters as above.
        # 6. Compute diagnostic variables
        # Reference ET using fixed surface resistance (70 s m-1 for grass)
        reference_surface_resistance = torch.full_like(
            surface_resistance,
            self._require_manager_param(
                "hydrosphere.physical",
                "reference_surface_resistance",
            ),
        )
        reference_canopy_conductance = 1.0 / reference_surface_resistance
        
        ref_resistance_factor = aerodynamic_conductance / reference_canopy_conductance
        # References: Allen et al. (1998), FAO-56, Eq. 3 with the reference surface.
        # Mathematical expression: D_ref = lambda (Delta + gamma (1 + g_a / g_ref)) with
        #   g_ref = 1 / r_ref; D_ref >= lambda (Delta + gamma) > 0 as for D above.
        # Parameter meanings: r_ref hydrosphere.physical.reference_surface_resistance; the
        #   other symbols as for D.
        # Units: D_ref [Pa K-1 J kg-1]; r_ref [s m-1]; g_ref [m s-1].
        # Domain/constraints: r_ref > 0 by its parameter bounds, so g_ref > 0; a non-finite
        #   or non-positive D_ref raises instead of being clamped.
        # Sign convention: D_ref is a positive magnitude; reference ET takes the sign of the
        #   numerator before its lower clamp.
        ref_denominator = self.latent_heat_vaporization * (
            slope_svp + psychrometric_constant * (1.0 + ref_resistance_factor)
        )
        if not bool(torch.all(torch.isfinite(ref_denominator) & (ref_denominator > 0.0))):
            raise ValueError(
                "Penman-Monteith reference denominator lambda (Delta + gamma (1 + g_a / g_ref)) "
                "must be finite and positive on the validated domain"
            )

        reference_et = torch.clamp(
            numerator / ref_denominator,
            min=self._require_manager_param(
                "hydrosphere.numerical",
                "evapotranspiration_flux_clamp_min",
            ),
        )

        # Equilibrium evapotranspiration (Priestley-Taylor alpha=1.0)
        equilibrium_et = torch.clamp(
            slope_svp * available_energy / 
            (self.latent_heat_vaporization * (slope_svp + psychrometric_constant)),
            min=zero_floor,
        )
        return {
            "net_radiation": net_radiation,
            "soil_heat_flux": soil_heat_flux,
            "available_energy": available_energy,
            "vapor_pressure_deficit": vapor_pressure_deficit,
            "aerodynamic_conductance": aerodynamic_conductance,
            "canopy_conductance": canopy_conductance,
            "slope_svp": slope_svp,
            "psychrometric_constant": psychrometric_constant,
            "evapotranspiration_flux": evapotranspiration_flux,
            "negative_et_projection_residual": negative_et_projection_residual,
            "reference_et": reference_et,
            "equilibrium_et": equilibrium_et,
        }

    def _daylight_gauss_law(self, combined_state: Dict[str, Any], dt: float) -> Dict[str, Tensor]:
        """Daily means of the Penman-Monteith law over the daylight Gauss samples."""
        # References: Monteith (1965); Allen et al. (1998), FAO-56 Eqs. 3, 11, 13 and
        #   45-46 (Chapter 3, soil heat flux "for hourly or shorter periods"); Goudriaan
        #   (1986), Agric. For. Meteorol. 38, 249-254; relic.scientific.atmosphere.
        #   daylight_quadrature (FAO-56 Eqs. 24, 25, 28, 34; SWAT+ Eq. 1:1.3.1 course).
        # Mathematical expression: R_n,k = R_n - S_n + S_n,k (sum_k W_k S_n,k = S_n, so
        #   sum_k W_k R_n,k = R_n); T_k the course of (T, T_max, T_min); e_s,k = e0(T_k);
        #   D_k = max(e0(T_k) - e_a, 0); g_c,k the per-sample canopy conductance (or the
        #   constant FAO-56 reference r_s); G_k = 0.1 R_n,k if R_n,k > 0, else
        #   0.5 R_n,k (Eqs. 45-46 with the classifier of the hourly support) for
        #   soil_heat_flux_source "fraction", G_k = G_state for "state"; ET_k, the
        #   reference and equilibrium ET_k of the law with their lower clamps at every
        #   sample; every returned quantity X = sum_k W_k X_k.
        # Parameter meanings: R_n the 24-hour mean net radiation of the radiation owner;
        #   S_n its net shortwave part (the daily mean net longwave R_n - S_n is held
        #   constant over the day); S_n,k the S_n redistributed by the solar elevation;
        #   e_a the daily actual vapour pressure; W_i = (N/24) w_i at the n daylight
        #   Gauss instants, W_night = 1 - N/24; wind speed, pressure and an optional r_a
        #   held constant over the day.
        # Units: R, S, G [W m-2]; T [K]; e, D [Pa]; g [m s-1]; ET [kg m-2 s-1].
        # Domain/constraints: dt = seconds_per_day; every input check of the law applies
        #   to every sample. Declared approximations: constant net longwave over the day;
        #   Eqs. 45-46 are FAO-56 fractions for a dense grass cover, applied to the sample
        #   instants of every surface as on the hourly support (their daily weighted sum
        #   is not forced to the Eq. 42 value 0); at a sample with D_k floored at 0 the
        #   law's e_a,k = e0(T_k) enters the air density.
        # Sign convention: as forward (R_n and G positive into the surface and soil).
        daylight_quadrature.require_daily_step(dt, "PenmanMonteithET")
        node_count, shortwave_shape = self._daylight_rule
        values: Dict[str, Tensor] = {}
        for key in self.DAYLIGHT_LAW_INPUTS:
            if combined_state.get(key) is None:
                raise ValueError(
                    f"Required state variable missing for Penman-Monteith ET (daylight_gauss): {key!r}"
                )
            values[key] = self._ensure_tensor(combined_state[key])
        if combined_state.get("aerodynamic_resistance") is not None:
            values["aerodynamic_resistance"] = self._ensure_tensor(combined_state["aerodynamic_resistance"])
        if self.soil_heat_flux_source == "state" and combined_state.get("soil_heat_flux") is not None:
            values["soil_heat_flux"] = self._ensure_tensor(combined_state["soil_heat_flux"])
        sample_inputs: Dict[str, Tensor] = {}
        if self.canopy_resistance_source == "state":
            key = self.DAYLIGHT_CONDUCTANCE_INPUT
            if combined_state.get(key) is None:
                raise ValueError(
                    f"Required canopy conductance missing for Penman-Monteith ET (daylight_gauss): {key!r} [m s-1]"
                )
            sample_inputs["canopy_conductance"] = self._ensure_tensor(combined_state[key])
        samples = daylight_quadrature.daylight_sample_state(
            values, sample_inputs, node_count=node_count, shortwave_shape=shortwave_shape,
            flux_keys=("shortwave_net",), temperature=True, vapor_pressure_deficit=True,
            dtype=torch.float32)
        support = samples.support
        sample_state: Dict[str, Any] = dict(samples.state)
        net_shortwave = sample_state.pop("shortwave_net")
        sample_state["net_radiation"] = (
            sample_state["net_radiation"] - support.constant(values["shortwave_net"]) + net_shortwave)
        sample_law = self._penman_monteith_law(sample_state, dt, "daylight_sample")
        return {key: support.daily_mean(value) for key, value in sample_law.items()}

    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Audit the open water-to-latent-heat boundary mapping.

        References:
            - Monteith (1965), evaporation and the surface energy budget.
            - Allen et al. (1998), FAO-56 Penman-Monteith formulation.
        Mathematical expression:
            R_lambda = max(|LE - lambda E| / max(|LE|, |lambda E|)).
        Parameter meanings:
            E is the outward evapotranspiration flux [kg m-2 s-1],
            lambda is latent heat of vaporization from the physical-constant
            SSOT [J kg-1], and LE is latent heat flux [W m-2]. A unit scale is
            used only for a double-zero denominator.
        Open-system route:
            Water and latent heat leave through coupled surface-atmosphere
            boundary ports. This constitutive audit does not claim a closed
            soil-water stock or complete surface-energy balance.
        """

        del input_state

        def require_tensor(key: str) -> Tensor:
            if key not in output_updates:
                raise KeyError(f"Missing conservation output '{key}'")
            value = output_updates[key]
            if isinstance(value, UpdateItem):
                value = value.value
            if not isinstance(value, Tensor):
                raise TypeError(
                    f"Penman-Monteith conservation output '{key}' "
                    "must be a torch.Tensor."
                )
            if value.numel() == 0:
                raise ValueError(
                    f"Penman-Monteith conservation output '{key}' "
                    "must be non-empty."
                )
            if not torch.is_floating_point(value):
                raise TypeError(
                    f"Penman-Monteith conservation output '{key}' "
                    "must be floating."
                )
            if not bool(torch.all(torch.isfinite(value))):
                raise ValueError(
                    f"Penman-Monteith conservation output '{key}' "
                    "must be finite."
                )
            if bool(torch.any(value < 0.0)):
                raise ValueError(
                    f"Penman-Monteith conservation output '{key}' "
                    "must be non-negative."
                )
            return value

        evapotranspiration_flux = require_tensor(
            "evapotranspiration_flux"
        )
        latent_heat_flux = require_tensor("latent_heat_flux")
        if evapotranspiration_flux.shape != latent_heat_flux.shape:
            raise ValueError(
                "Penman-Monteith conservation outputs must have identical shapes."
            )
        if (
            evapotranspiration_flux.device != latent_heat_flux.device
            or evapotranspiration_flux.dtype != latent_heat_flux.dtype
        ):
            raise ValueError(
                "Penman-Monteith conservation outputs must share dtype and device."
            )

        expected_latent_heat = (
            evapotranspiration_flux
            * evapotranspiration_flux.new_tensor(
                float(self.latent_heat_vaporization)
            )
        )
        scale = torch.maximum(
            torch.abs(latent_heat_flux),
            torch.abs(expected_latent_heat),
        )
        scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
        residual = torch.abs(latent_heat_flux - expected_latent_heat) / scale
        return {
            "latent_heat_boundary": float(
                torch.max(residual).detach().item()
            )
        }
    

    def _es_tetens_pa(self, temp_K: Tensor) -> Tensor:
        """Compute saturation vapor pressure using the shared psychrometrics helper."""
        # References:
        #   Allen et al. (1998), FAO-56, equations 11 and 13.
        # Mathematical expression:
        #   Tc=T-K0; es=e0*exp(a*Tc/(Tc+b)); Delta=c*es/(Tc+b)^2.
        # Parameter meanings:
        #   e0,a,b,c and K0 are the shared psychrometric physical constants;
        #   T is absolute air temperature. These wrappers delegate to the shared owner.
        # Units:
        #   T/K0 [K], Tc/b [degC interval], es/e0 [Pa], Delta [Pa K-1].
        # Domain/constraints:
        #   Use the configured valid temperature interval; do not create a second
        #   Tetens constant table or apply an absolute-temperature offset to a difference.
        # Sign convention:
        #   es and Delta are nonnegative meteorological properties, not signed fluxes.
        return compute_saturation_vapor_pressure(
            temp_K,
            tetens_base_pressure=self._tetens_base_pressure,
            tetens_coeff_a=self._tetens_coeff_a,
            tetens_coeff_b=self._tetens_coeff_b,
            kelvin_offset=self._celsius_to_kelvin,
        )
    
    def _daily_saturation_vapor_pressure(self, state: Mapping[str, Any]) -> Tensor:
        """Daily mean saturation vapour pressure of FAO-56 Eq. 12 [Pa]."""
        # References: Allen et al. (1998), FAO-56, Eqs. 11 and 12.
        # Mathematical expression: e_s = (e0(T_max) + e0(T_min)) / 2 with
        #   e0(T) = e0_ref exp(a Tc / (Tc + b)) the Tetens form of Eq. 11.
        # Parameter meanings: T_max and T_min the daily maximum and minimum air
        #   temperature; e0_ref, a, b the shared Tetens constants (_es_tetens_pa).
        # Units: T_max, T_min [K]; e_s, e0 [Pa].
        # Domain/constraints: both inputs required on the daily support, finite, inside
        #   hydrosphere.numerical.temperature_bounds and T_min <= T_max. FAO-56 states that
        #   e0 at the daily mean temperature instead of this mean underestimates e_s
        #   (the convexity of Eq. 11), hence the vapour pressure deficit and ET.
        # Sign convention: e_s is a nonnegative meteorological property.
        extrema = {}
        for key in self.DAILY_TEMPERATURE_INPUTS:
            if state.get(key) is None:
                raise ValueError(
                    f"Required state variable missing for daily Penman-Monteith ET: {key!r}"
                )
            extrema[key] = self._ensure_tensor(state[key])  # K
        temp_bounds = self.get_parameter_value(
            "temperature_bounds",
            namespace="hydrosphere.numerical", required=True)
        for value in extrema.values():
            validate_temperature_range(
                value, min_temp=float(temp_bounds[0]), max_temp=float(temp_bounds[1])
            )
        t_max = extrema["air_temperature_max"]
        t_min = extrema["air_temperature_min"]
        if bool(torch.any(t_min > t_max)):
            raise ValueError("air_temperature_min must not exceed air_temperature_max")
        return 0.5 * (self._es_tetens_pa(t_max) + self._es_tetens_pa(t_min))

    def _slope_svp_pa_per_K(self, temp_K: Tensor) -> Tensor:
        """Compute slope of saturation vapor pressure using the shared helper."""
        # References:
        #   Allen et al. (1998), FAO-56, equations 11 and 13.
        # Mathematical expression:
        #   Tc=T-K0; es=e0*exp(a*Tc/(Tc+b)); Delta=c*es/(Tc+b)^2.
        # Parameter meanings:
        #   e0,a,b,c and K0 are the shared psychrometric physical constants;
        #   T is absolute air temperature. These wrappers delegate to the shared owner.
        # Units:
        #   T/K0 [K], Tc/b [degC interval], es/e0 [Pa], Delta [Pa K-1].
        # Domain/constraints:
        #   Use the configured valid temperature interval; do not create a second
        #   Tetens constant table or apply an absolute-temperature offset to a difference.
        # Sign convention:
        #   es and Delta are nonnegative meteorological properties, not signed fluxes.
        return compute_slope_saturation_vapor_pressure(
            temp_K,
            tetens_base_pressure=self._tetens_base_pressure,
            tetens_coeff_a=self._tetens_coeff_a,
            tetens_coeff_b=self._tetens_coeff_b,
            tetens_slope_coeff=self._tetens_slope_coeff,
            kelvin_offset=self._celsius_to_kelvin,
        )
    
    def _compute_aerodynamic_conductance(self, 
                                       wind_speed: Tensor, 
                                       air_temp: Tensor) -> Tensor:
        """
        Compute aerodynamic conductance using logarithmic wind profile.
        
        Mathematical expression:
            ga = k2 * u / [ln((z-d)/z0m) * ln((z-d)/z0h)]
        
        Parameters
        ----------
        wind_speed : Tensor
            Wind speed at reference height [m s-1]
        air_temp : Tensor
            Air temperature for stability corrections [K]
            
        Returns
        -------
        Tensor
            Aerodynamic conductance [m s-1]
        """
        # References:
        #   Allen et al. (1998), FAO-56; Monteith and Unsworth (2013).
        # Mathematical expression:
        #   ga = kappa^2*max(u,u_min) / [ln((z-d)/z0m)*ln((z-d)/z0h)].
        # Parameter meanings:
        #   u is reference wind; z-d is effective height; z0m/z0h are roughness lengths;
        #   kappa is the physical von Karman constant.
        # Units:
        #   ga/u [m s-1]; heights/roughness [m]; logarithms and kappa [1].
        # Domain/constraints:
        #   Neutral log-profile approximation only; positive roughness and effective
        #   height above roughness; apply configured conductance bounds. air_temp is not
        #   used here to evaluate a stability correction.
        # Sign convention:
        #   Positive ga denotes a transfer magnitude; PM supplies the vapor-gradient sign.
        # The law and its validation are owned by fao56_aerodynamic_conductance; this
        # method reads the parameters in their former order and delegates.
        minimum_wind_speed = self._require_manager_param(
            "hydrosphere.numerical",
            "minimum_wind_speed",
        )
        heat_to_momentum_ratio = self._require_manager_param(
            "hydrosphere.physical",
            "heat_to_momentum_roughness_ratio",
        )
        ga_bounds = self._require_manager_param(
            "hydrosphere.numerical",
            "aerodynamic_conductance_bounds",
        )
        return fao56_aerodynamic_conductance(
            wind_speed,
            reference_height=self.reference_height,
            displacement_height=self.displacement_height,
            roughness_length=self.roughness_length,
            heat_to_momentum_ratio=heat_to_momentum_ratio,
            von_karman_constant=self.von_karman_constant,
            minimum_wind_speed=minimum_wind_speed,
            conductance_bounds=ga_bounds,
        )
    
    def _require_manager_param(
        self,
        namespace: str,
        key: str,
    ) -> Any:
        """Fetch a mandatory scalar or object from ParameterManager namespace."""
        return self.get_parameter_value(
            key,
            namespace=namespace,
            source_order=("manager",), required=True)

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
        return super()._ensure_tensor(value, dtype=torch.float32)

    def _process_canopy_interception(self, 
                                   total_et: Tensor,
                                   state: StateDict,
                                   dt: float) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Process canopy interception and separate evaporation/transpiration.
        
        This method implements a capacity-overflow canopy store (the bucket form of
        the Rutter et al. 1971 / Gash 1979 store used by land surface models, e.g.
        CLM4.5, Oleson et al. 2013, Sect. 7.1, and RHESSys), where:
        1. Precipitation first fills canopy storage up to capacity
        2. Evaporation preferentially occurs from intercepted water (wet canopy)
        3. Remaining ET demand is satisfied by transpiration (dry canopy)
        4. Only the excess storage above capacity drains (overflow drip) and joins
           throughfall; no exponential Rutter drainage D = D_s exp[b (C - S)] below
           capacity is executed, so no drainage coefficient b is read
        
        IMPORTANT: This method is STATELESS - it reads canopy_storage from state
        and returns the NEW storage value, not modifying any internal state.
        
        Mathematical Expression
        -----------------------
        C_max = capacity_per_LAI * LAI
        I = min(P, max(0, C_max - S_old))
        S_new = S_old + I - E_i
        T_fall = P - I + max(0, S_new - C_max)
        
        where:
        - C_max: Maximum canopy storage capacity [mm]
        - I: Interception [mm]
        - P: Precipitation [mm] 
        - S: Canopy storage [mm]
        - E_i: Evaporation from interception [mm]
        - T_fall: Throughfall [mm]
        
        Parameters
        ----------
        total_et : Tensor
            Total ET demand from Penman-Monteith [kg m-2 s-1]
        state : StateDict
            System state including precipitation, LAI, and canopy_storage
        dt : float
            Time step [s]
            
        Returns
        -------
        tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
            - adjusted_total_et: Total ET flux (same as input, for mass balance)
            - evap_from_interception: Evaporation from canopy storage
            - transpiration_flux: Plant transpiration flux
            - throughfall_flux: Net precipitation reaching soil
            - storage_change: Change in canopy storage over the step, S_new - S_old [mm]
            - new_canopy_storage: Updated canopy storage value (NEW - for StateManager)
        """
        # Canopy interception requires its canopy inputs; none is defaulted.
        missing = [
            name
            for name in ("precipitation_flux", "canopy_storage", "leaf_area_index")
            if state.get(name) is None
        ]
        if missing:
            raise ValueError(
                "Canopy interception in Penman-Monteith ET requires "
                + ", ".join(repr(name) for name in missing)
            )
        precipitation_flux = self._ensure_tensor(state['precipitation_flux'])  # kg m-2 s-1
        leaf_area_index = self._ensure_tensor(state['leaf_area_index'])  # dimensionless

        if dt <= 0.0:
            raise ValueError("Time step dt must be positive for canopy interception calculations.")

        # Read current canopy storage from state (NOT from self)
        current_canopy_storage = self._ensure_tensor(state['canopy_storage']).to(
            device=precipitation_flux.device, dtype=precipitation_flux.dtype
        )  # mm
        
        # Convert rates to per-step depths: 1 kg m-2 of water equals 1 mm.
        dt_tensor = torch.as_tensor(
            dt, device=precipitation_flux.device, dtype=precipitation_flux.dtype
        )
        precipitation_mm = precipitation_flux * dt_tensor  # 1 kg m-2 == 1 mm
        total_et_mm = total_et * dt_tensor  # mm
        
        # 1. Calculate canopy storage capacity
        max_capacity = (
            self._ensure_tensor(self.interception_capacity_per_LAI).to(
                device=leaf_area_index.device, dtype=leaf_area_index.dtype
            )
            * leaf_area_index
        )  # mm
        canopy_storage_bounds = self._require_manager_param(
            "hydrosphere.numerical",
            "canopy_storage_bounds",
        )
        max_capacity = torch.clamp(
            max_capacity, min=canopy_storage_bounds[0], max=canopy_storage_bounds[1]
        )  # Physical bounds
        
        # 2. Process precipitation interception
        # Available capacity for new interception
        available_capacity = torch.clamp(
            max_capacity - current_canopy_storage, min=0.0
        )
        
        # Actual interception (limited by available capacity and precipitation)
        interception = torch.minimum(precipitation_mm, available_capacity)
        
        # Throughfall: precipitation not intercepted
        direct_throughfall = precipitation_mm - interception
        
        # Calculate new canopy storage with new interception (no self mutation)
        new_canopy_storage = current_canopy_storage + interception
        
        # 3. Process evaporation from canopy (priority over transpiration)
        # Enhanced evaporation efficiency for wet canopy
        potential_evap_from_canopy = (
            total_et_mm * self._ensure_tensor(self.canopy_evaporation_efficiency).to(
                device=total_et_mm.device, dtype=total_et_mm.dtype
            )
        )
        
        # Actual evaporation limited by available storage
        evap_from_interception_mm = torch.minimum(
            new_canopy_storage, potential_evap_from_canopy
        )
        
        # Update canopy storage after evaporation (still local variable)
        new_canopy_storage -= evap_from_interception_mm
        
        # 4. Process drip/overflow from excess storage
        drip = torch.clamp(new_canopy_storage - max_capacity, min=0.0)
        new_canopy_storage = torch.minimum(new_canopy_storage, max_capacity)
        direct_throughfall = direct_throughfall + drip
        
        # 5. Remaining ET demand becomes transpiration
        transpiration_mm = torch.clamp(
            total_et_mm - evap_from_interception_mm, min=0.0
        )
        
        # 6. Convert back to flux units (kg m-2 s-1 == mm s-1)
        evap_from_interception_flux = evap_from_interception_mm / dt_tensor
        transpiration_flux = transpiration_mm / dt_tensor
        throughfall_flux = direct_throughfall / dt_tensor
        
        # References: Rutter et al. (1971); Gash (1979).
        # Mathematical expression: Delta S = S_new - S_old = I - E_i - Drip;
        #   S_old + P = S_new + E_i + T_fall with T_fall = (P - I) + Drip.
        # Parameter meanings: S_old caller storage, S_new returned storage, I
        #   interception, E_i interception evaporation, Drip capacity overflow,
        #   P precipitation depth, T_fall throughfall depth over the step.
        # Units: every depth is mm of liquid water per step; dt [s].
        # Domain/constraints: availability for E_i is S_old + I, the store
        #   present before evaporation; the returned storage is clipped to
        #   capacity only through the explicit drip term.
        # Sign convention: negative Delta S drains the canopy store; drip and
        #   evaporation are nonnegative magnitudes.
        storage_change = new_canopy_storage - current_canopy_storage
        
        # Mass balance conservation checks and diagnostics
        self._validate_canopy_mass_balance(
            precipitation_mm, interception, evap_from_interception_mm,
            direct_throughfall, total_et_mm, transpiration_mm, dt,
            storage_before_mm=current_canopy_storage,
            storage_after_mm=new_canopy_storage,
        )
        
        # Ensure total ET is conserved (evaporation + transpiration = total)
        adjusted_total_et = evap_from_interception_flux + transpiration_flux
        
        return (
            adjusted_total_et,
            evap_from_interception_flux,
            transpiration_flux,
            throughfall_flux,
            storage_change,
            new_canopy_storage  # NEW: Return updated storage for StateManager
        )


    def _validate_canopy_mass_balance(self,
                                    precipitation_mm: Union[float, Tensor],
                                    interception_mm: Union[float, Tensor],
                                    evap_from_interception_mm: Union[float, Tensor],
                                    throughfall_mm: Union[float, Tensor],
                                    total_et_mm: Union[float, Tensor],
                                    transpiration_mm: Union[float, Tensor],
                                    dt: float,
                                    storage_before_mm: Union[float, Tensor],
                                    storage_after_mm: Union[float, Tensor]) -> None:
        """Validate the canopy interception water budget at consistent time levels.

        References:
          - Rutter et al. (1971); Gash (1979), canopy interception storage models.
          - Allen et al. (1998), FAO-56 evapotranspiration partitioning.
        Mathematical expression:
          S_old + P = S_new + E_i + T_fall (complete canopy water budget);
          T_fall = (P - I) + Drip with Drip >= 0 (precipitation partition);
          ET_total = E_i + T (evapotranspiration partition);
          0 <= I <= P; 0 <= E_i <= S_old + I; T >= 0; S_old, S_new >= 0.
        Parameter meanings:
          P precipitation depth, I interception, E_i evaporation from the
          intercepted store, T_fall throughfall including drip, ET_total the
          Penman-Monteith demand, T transpiration, S_old the storage before the
          step and S_new the storage returned for StateManager.
        Units:
          every depth is in mm of liquid water over the step; dt [s] is only
          forwarded to the diagnostics logger; tolerances are in mm.
        Domain/constraints:
          the availability check uses S_old + I, the store present before
          evaporation, not the depleted S_new; residuals above the manager
          tolerance warn and above the critical threshold fail.
        Sign convention:
          all depths are nonnegative magnitudes; the budget residual is
          (S_old + P) - (S_new + E_i + T_fall) and closes at zero.
        """
        tolerance_mm = float(self._require_manager_param(
            "hydrosphere.numerical",
            "mass_conservation_tolerance_mm",
        ))
        critical_error_threshold = float(self._require_manager_param(
            "hydrosphere.numerical",
            "critical_mass_balance_error_threshold",
        ))

        def _scalar_max(value: Tensor) -> float:
            if value.numel() == 1:
                return self._scalar_from_value(value)
            return self._scalar_from_value(torch.max(value))

        precipitation_t = self._ensure_tensor(precipitation_mm)
        interception_t = self._ensure_tensor(interception_mm)
        throughfall_t = self._ensure_tensor(throughfall_mm)
        evap_t = self._ensure_tensor(evap_from_interception_mm)
        total_et_t = self._ensure_tensor(total_et_mm)
        transpiration_t = self._ensure_tensor(transpiration_mm)
        storage_before_t = self._ensure_tensor(storage_before_mm)
        storage_after_t = self._ensure_tensor(storage_after_mm)

        # 1. Storage sign at both time levels.
        if torch.any(storage_before_t < -tolerance_mm):
            raise AssertionError(
                f"CRITICAL: Negative canopy storage before the step: {_scalar_max(storage_before_t):.6f} mm"
            )
        if torch.any(storage_after_t < -tolerance_mm):
            raise AssertionError(
                f"CRITICAL: Negative canopy storage: {_scalar_max(storage_after_t):.6f} mm"
            )

        # 2. Precipitation partition: 0 <= I <= P and T_fall = (P - I) + Drip with Drip >= 0.
        if torch.any(interception_t > precipitation_t + tolerance_mm):
            raise AssertionError(
                f"CRITICAL: Precipitation mass balance violation: interception exceeds precipitation: "
                f"I={_scalar_max(interception_t):.6f} mm > P={_scalar_max(precipitation_t):.6f} mm"
            )
        direct_throughfall_t = precipitation_t - interception_t
        drip_t = throughfall_t - direct_throughfall_t
        if torch.any(drip_t < -tolerance_mm):
            raise AssertionError(
                f"CRITICAL: Precipitation mass balance violation: throughfall below direct throughfall: "
                f"T_fall={_scalar_max(throughfall_t):.6f} mm < P-I={_scalar_max(direct_throughfall_t):.6f} mm"
            )

        # 3. Evaporation availability uses the store present before evaporation.
        available_storage = storage_before_t + interception_t
        if torch.any(evap_t > available_storage + tolerance_mm):
            raise AssertionError(
                f"CRITICAL: Evaporation exceeds available storage: "
                f"E_i={_scalar_max(evap_t):.6f} mm > "
                f"Available={_scalar_max(available_storage):.6f} mm"
            )

        if torch.any(transpiration_t < -tolerance_mm):
            raise AssertionError(
                f"CRITICAL: Negative transpiration: T={_scalar_max(transpiration_t):.6f} mm"
            )

        # 4. Evapotranspiration partition: ET_total = E_i + T.
        et_balance = total_et_t - (evap_t + transpiration_t)
        et_balance_abs = torch.max(torch.abs(et_balance))
        if et_balance_abs > tolerance_mm:
            error_msg = (
                f"ET mass balance violation: "
                f"ET_total={_scalar_max(total_et_t):.6f} mm, E_i={_scalar_max(evap_t):.6f} mm, "
                f"T={_scalar_max(transpiration_t):.6f} mm, Error={_scalar_max(torch.abs(et_balance)):.6e} mm"
            )
            if et_balance_abs > critical_error_threshold:
                raise AssertionError(f"CRITICAL: {error_msg}")
            else:
                warnings.warn(f"WARNING: {error_msg}")

        # 5. Complete canopy water budget: S_old + P = S_new + E_i + T_fall.
        budget_residual = (storage_before_t + precipitation_t) - (
            storage_after_t + evap_t + throughfall_t
        )
        budget_residual_abs = torch.max(torch.abs(budget_residual))
        if budget_residual_abs > tolerance_mm:
            error_msg = (
                f"Canopy water budget violation: "
                f"S_old={_scalar_max(storage_before_t):.6f} mm, P={_scalar_max(precipitation_t):.6f} mm, "
                f"S_new={_scalar_max(storage_after_t):.6f} mm, E_i={_scalar_max(evap_t):.6f} mm, "
                f"T_fall={_scalar_max(throughfall_t):.6f} mm, "
                f"Residual={_scalar_max(torch.abs(budget_residual)):.6e} mm"
            )
            if budget_residual_abs > critical_error_threshold:
                raise AssertionError(f"CRITICAL: {error_msg}")
            else:
                warnings.warn(f"WARNING: {error_msg}")

        # 6. Canopy storage bounds check on the returned storage.
        max_lai_factor = float(self._require_manager_param(
            "hydrosphere.physical",
            "maximum_lai_factor",
        ))
        max_physical_capacity = self.interception_capacity_per_LAI * max_lai_factor
        if torch.any(storage_after_t > max_physical_capacity):
            warnings.warn(
                f"Canopy storage exceeds physical bounds: "
                f"Storage={_scalar_max(storage_after_t):.3f} mm > Max={max_physical_capacity:.3f} mm"
            )

        # 7. Diagnostic reporting (if enabled via feature flag)
        if feature_manager.is_enabled('canopy_interception_diagnostics'):
            self._log_canopy_diagnostics(
                _scalar_max(precipitation_t),
                _scalar_max(interception_t),
                _scalar_max(evap_t),
                _scalar_max(throughfall_t),
                _scalar_max(total_et_t),
                _scalar_max(transpiration_t),
                _scalar_max(budget_residual),
                _scalar_max(et_balance),
                dt,
                _scalar_max(storage_after_t),
            )

    def _log_canopy_diagnostics(self,
                              precipitation_mm: float,
                              interception_mm: float,
                              evap_from_interception_mm: float, 
                              throughfall_mm: float,
                              total_et_mm: float,
                              transpiration_mm: float,
                              budget_residual: float,
                              et_balance: float,
                              dt: float,
                              canopy_storage: float) -> None:
        """Log detailed canopy interception diagnostics for debugging and analysis."""
        import logging
        logger = logging.getLogger(__name__ + '.canopy_diagnostics')
        
        # Calculate rates and ratios for diagnostic purposes
        dt_hours = dt / self._seconds_per_hour
        interception_rate = interception_mm / dt_hours if dt_hours > 0 else 0  # mm/h
        evap_rate = evap_from_interception_mm / dt_hours if dt_hours > 0 else 0  # mm/h
        throughfall_rate = throughfall_mm / dt_hours if dt_hours > 0 else 0  # mm/h
        
        # Interception efficiency
        # References: CF Standard Name Table v94 ("lwe" means liquid water equivalent);
        #   UDUNITS-2; Rutter et al. (1971).
        # Mathematical expression: eff = 100 I / P when P > P_min, otherwise 0, with
        #   P_min = U(F_min; kg m-2 s-1 -> mm s-1) dt.
        # Parameter meanings: I interception depth and P precipitation depth of the step;
        #   F_min hydrosphere.numerical.minimum_drainage_flux; U the UnitConverter
        #   liquid-water-equivalent rule (rho_w = relic.physical.constants.water.density_liquid);
        #   dt the step length.
        # Units: I, P, P_min [mm]; F_min [kg m-2 s-1]; U(F_min) [mm s-1]; dt [s]; eff [%].
        # Domain/constraints: dt > 0 (the interception step rejects dt <= 0); both sides of
        #   the comparison are depths in mm; the value is only logged, never fed back.
        # Sign convention: depths are non-negative magnitudes; eff = 0 at or below P_min.
        minimum_precipitation_mm = (
            float(
                unit_converter.convert(
                    self._require_manager_param(
                        "hydrosphere.numerical",
                        "minimum_drainage_flux",
                    ),
                    "drainage_flux",
                    "kg m-2 s-1",
                    "mm s-1",
                )
            )
            * dt
        )
        interception_efficiency = (
            interception_mm / precipitation_mm * 100
            if precipitation_mm > minimum_precipitation_mm
            else 0
        )  # %

        # Wet canopy fraction (proxy based on storage)
        # References: Rutter et al. (1971); Gash (1979)
        # Expression: S_cap = C_LAI * L_mod, with L_mod representing a climatological LAI
        # Parameters: hydrosphere.physical.moderate_lai_factor (m2 m-2) defines L_mod
        moderate_lai_factor = self._require_manager_param(
            "hydrosphere.physical",
            "moderate_lai_factor",
        )
        max_capacity = self.interception_capacity_per_LAI * moderate_lai_factor  # Assume moderate LAI
        wet_fraction = min(1.0, canopy_storage / max_capacity) if max_capacity > 0 else 0
        
        logger.debug(f"Canopy Interception Diagnostics (dt={dt:.1f}s):")
        logger.debug(f"  Input: P={precipitation_mm:.3f} mm ({precipitation_mm/dt_hours:.3f} mm/h)")
        logger.debug(f"  Interception: I={interception_mm:.3f} mm ({interception_rate:.3f} mm/h)")
        logger.debug(f"  Throughfall: T_fall={throughfall_mm:.3f} mm ({throughfall_rate:.3f} mm/h)")
        logger.debug(f"  Evaporation from interception: E_i={evap_from_interception_mm:.3f} mm ({evap_rate:.3f} mm/h)")
        logger.debug(f"  Transpiration: T={transpiration_mm:.3f} mm")
        logger.debug(f"  Total ET: ET={total_et_mm:.3f} mm")
        logger.debug(f"  Canopy storage: S={canopy_storage:.3f} mm")
        logger.debug(f"  Storage capacity: C_max={max_capacity:.3f} mm")
        logger.debug(f"  Interception efficiency: {interception_efficiency:.1f}%")
        logger.debug(f"  Wet canopy fraction: {wet_fraction:.3f}")
        logger.debug("  Mass balance errors:")
        logger.debug(f"    Canopy water budget residual: {budget_residual:.2e} mm")
        logger.debug(f"    ET balance: {et_balance:.2e} mm")


# Export the main class
__all__ = ["PenmanMonteithET"]
