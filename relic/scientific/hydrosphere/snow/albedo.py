"""
Snow Albedo Dynamics Module for RELIC

This module implements dynamic snow albedo parameterizations based on
BATS (Biosphere-Atmosphere Transfer Scheme) and simple age-based models.
It accounts for snow aging, temperature effects, and fresh snowfall events.

References
----------
- Dickinson, R.E., Henderson-Sellers, A., Kennedy, P.J. (1993). 
  Biosphere-Atmosphere Transfer Scheme (BATS) Version 1e as Coupled 
  to the NCAR Community Climate Model. NCAR Technical Note.
- Molotch, N.P., Bales, R.C. (2006). SNTHERM model validation using 
  laboratory experiments. Journal of Geophysical Research, 111, D23105.
- RHESSys snowpack_daily_F.c implementation

Mathematical Expressions
------------------------
BATS Model Nondimensional Snow Age Evolution (state snow_nondimensional_age):
    tau_s = tau_s_start + dt_seconds * decay_rate(T) * (r1 + r2 + r3)
    where decay_rate is albedo_decay_rate_cold or albedo_decay_rate_warm [s-1] and
        r1 = exp(bats_grain_growth_temperature_factor *
                 (1 / bats_reference_temperature - 1 / (T_surface + bats_temperature_epsilon)))
        where all parameters are loaded from hydrosphere.snow namespace; BATS 1e and the
        RHESSys reference code use r1 = exp(5000 K (1/273.16 K - 1/T)) with no extra multiplier
        r2 = min(r1^10, 1.0) # meltwater refreeze effect  
        r3 = dust_soot_factor # dust/soot contamination (default 0.3)

BATS Albedo Calculation:
    fage = tau_s / (1 + tau_s)
    albedo_vis = bats_albedo_vis_base * (1 - bats_albedo_vis_age_factor * fage)
                 + bats_albedo_zenith_factor * fz * (1 - bats_albedo_vis_base * (1 - bats_albedo_vis_age_factor * fage))
    albedo_nir = bats_albedo_nir_base * (1 - bats_albedo_nir_age_factor * fage)
                 + bats_albedo_zenith_factor * fz * (1 - bats_albedo_nir_base * (1 - bats_albedo_nir_age_factor * fage))
    surface_albedo = bats_albedo_average_weight * albedo_vis + (1 - bats_albedo_average_weight) * albedo_nir

Solar Zenith Angle Effect:
    if cos(theta_noon) < 0.5:
        fz = (1/b) * ((1+b) / (1 + 2*b*cos(theta_noon)) - 1.0)
    else:
        fz = 0.0

Fresh Snow Reset:
    if snowfall > snow_reset_threshold:
        surface_age = 0.0
        
Units Convention (CF-1.8):
    surface_albedo: dimensionless (0-1)
    snowfall: kg m-2 s-1 
    air_temperature: K
    snow_surface_temperature: K
"""

from typing import Any, Dict, List, Mapping, Optional, Union
import torch
from torch import Tensor

from relic.computation.operators.dec.forms import CochainSpaceSpec
from relic.computation.operators.dec.hodge import HodgeStarSpec
from relic.computation.operators.iphs import PortSpecIPHS
from relic.core.abstractions import StateDict, FluxDict
from relic.core.functor import (
    DependencyDeclaration,
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
    SpatialScale,
    VariableKind,
    TemporalScale,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant
from ..interface_mapping import HydrosphereInterfaceMixin

numeric_types = (float, int, Tensor)

class SnowAlbedo(HydrosphereInterfaceMixin, PhysicalFieldFunctor):
    """
    Snow Albedo Dynamics Model
    
    Implements dynamic snow surface albedo evolution based on snow age,
    temperature effects, and fresh snowfall events. Supports both BATS
    and simple age-based parameterizations for integration with radiation
    and energy balance modules.
    
    The model tracks snow surface age and computes spectrally-averaged
    albedo accounting for:
    - Snow grain metamorphism and aging
    - Temperature-dependent aging rates  
    - Solar zenith angle effects
    - Fresh snowfall reset events
    - Dust and soot contamination effects
    
    Attributes
    ----------
    albedo_scheme : str
        Albedo parameterization scheme ('bats' or 'age_based')
    fresh_snow_albedo : float
        Albedo of fresh snow [dimensionless]
    old_snow_albedo : float 
        Albedo of aged snow [dimensionless]
    snow_reset_threshold : float
        Snowfall threshold for age reset [kg m-2 s-1]
    bats_b_parameter : float
        BATS solar zenith angle factor [dimensionless]
    dust_soot_factor : float
        BATS contamination factor [dimensionless]
    aging_time_scale : float
        Characteristic aging time scale [days]
    """

    PHASE8_RUNTIME_REPLACEMENT_STATUS = "canonical_iphs_runtime"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None
    
    def __init__(self, 
                 name: str = "snow_albedo",
                 config: Optional[Dict] = None) -> None:
        """
        Initialize Snow Albedo model.
        
        Parameters
        ----------
        name : str, optional
            Unique identifier for this functor instance
        config : Dict, optional
            Configuration parameters for albedo scheme
        """
        # Create metadata
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Dynamic snow surface albedo based on BATS and age-based parameterizations",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "BATS: alpha_s = f(age, T_surface, solar_zenith); "
                "Age: alpha_s = alpha_old + (alpha_fresh-alpha_old) exp(-age/tau); "
                "q_rad = star_alpha I_sw with star_alpha = diag(1-alpha_s)"
            ),
            references=[
                "Dickinson, R.E. et al. (1993). BATS Version 1e. NCAR Technical Note",
                "Molotch, N.P., Bales, R.C. (2006). SNTHERM validation. JGR 111, D23105",
                "RHESSys snowpack_daily_F.c implementation"
            ],
            parameters={
                "hydrosphere.numerical.epsilon_small": (
                    "Floor of the absorbed fraction of the static absorption Hodge contract "
                    "[dimensionless]"
                ),
                "hydrosphere.snow.albedo_scheme": (
                    "Snow albedo parameterization scheme (bats or age_based) [dimensionless]"
                ),
                "hydrosphere.snow.fresh_snow_albedo": (
                    "Albedo of fresh snow (visible spectrum) [dimensionless]"
                ),
                "hydrosphere.snow.old_snow_albedo": (
                    "Albedo of old snow (lower bound for age-based model) [dimensionless]"
                ),
                "hydrosphere.snow.bare_ground_albedo_factor": (
                    "Multiplicative factor mapping old snow albedo to bare-ground fallback albedo "
                    "[dimensionless]"
                ),
                "hydrosphere.snow.snow_reset_threshold": (
                    "Snowfall threshold for resetting snow age to zero [kg m-2 s-1]"
                ),
                "hydrosphere.snow.bats_b_parameter": (
                    "BATS solar zenith angle factor for albedo calculation [dimensionless]"
                ),
                "hydrosphere.snow.dust_soot_factor": (
                    "BATS dust and soot contamination factor for snow aging [dimensionless]"
                ),
                "hydrosphere.snow.albedo_decay_rate_cold": (
                    "Albedo decay rate for cold conditions (T < 0 degC), tau_cold = 10 days = "
                    "1/(1.16e-6 s-1) [s-1]"
                ),
                "hydrosphere.snow.albedo_decay_rate_warm": (
                    "Albedo decay rate for warm conditions (T > 0 degC), tau_warm = 1 day = "
                    "1/(1.16e-5 s-1) [s-1]"
                ),
                "hydrosphere.snow.temperature_threshold_warm": (
                    "Temperature threshold for switching from cold to warm albedo decay (0 degC = "
                    "273.15 K) [K]"
                ),
                "hydrosphere.snow.bats_grain_growth_temperature_factor": (
                    "BATS grain growth temperature factor b in r1 = exp(b (1/T_ref - 1/T)) [K]"
                ),
                "hydrosphere.snow.age_based_temperature_sensitivity": (
                    "Warm-snow aging sensitivity kappa of the age-based albedo scheme [K-1]"
                ),
                "hydrosphere.snow.bats_reference_temperature": (
                    "BATS reference temperature for grain growth exponent [K]"
                ),
                "hydrosphere.snow.bats_temperature_epsilon": (
                    "Small temperature offset to avoid division by zero in BATS r1 term [K]"
                ),
                "hydrosphere.snow.bats_cos_zenith_threshold": (
                    "Cosine zenith threshold for BATS fz adjustment [dimensionless]"
                ),
                "hydrosphere.snow.bats_albedo_vis_base": (
                    "BATS visible-band fresh snow albedo coefficient [dimensionless]"
                ),
                "hydrosphere.snow.bats_albedo_vis_age_factor": (
                    "BATS visible-band aging factor applied to fage [dimensionless]"
                ),
                "hydrosphere.snow.bats_albedo_nir_base": (
                    "BATS near-infrared fresh snow albedo coefficient [dimensionless]"
                ),
                "hydrosphere.snow.bats_albedo_nir_age_factor": (
                    "BATS near-infrared aging factor applied to fage [dimensionless]"
                ),
                "hydrosphere.snow.bats_albedo_zenith_factor": (
                    "BATS solar zenith albedo adjustment factor for VIS and NIR [dimensionless]"
                ),
                "hydrosphere.snow.bats_albedo_average_weight": (
                    "Weight for combining VIS and NIR albedo components [dimensionless]"
                ),
                "hydrosphere.snow.aging_time_scale": (
                    "Characteristic time scale for age-based albedo decay model [day]"
                ),
                "hydrosphere.snow.minimum_snow_depth": (
                    "Minimum snow depth threshold for albedo calculation [m]"
                ),
                "hydrosphere.snow.albedo_clamp_min": (
                    "Minimum allowed surface albedo value [dimensionless]"
                ),
                "hydrosphere.snow.albedo_clamp_max": (
                    "Maximum allowed surface albedo value [dimensionless]"
                ),
                "hydrosphere.snow.hodge_reference_surface_albedo": (
                    "Reference aged-snow albedo of the static absorption Hodge contract "
                    "[dimensionless]"
                ),
                "hydrosphere.snow.temperature_clamp_min": (
                    "Minimum temperature for BATS aging calculation (-50 degC) [K]"
                ),
                "hydrosphere.snow.temperature_clamp_max": (
                    "Maximum temperature for BATS aging calculation (+10 degC) [K]"
                ),
            },
            input_variables=[
                "snowfall", "snow_surface_temperature", "air_temperature",
                "solar_zenith_angle", "snow_depth", "shortwave_down"
            ],
            output_variables=[
                "surface_albedo",
                "snow_surface_age",
                "snow_nondimensional_age",
                "diagnostics.snow_albedo",
            ],
            optional_output_variables=["snow_surface_age", "snow_nondimensional_age"],
            conservation_laws=["energy"],
            spatial_dimensions=0,  # Point-based model
            temporal_scale=TemporalScale.DAY,
            coupling_interfaces=["radiation", "energy_balance", "snow_physics"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        
        # Initialize parent class
        base_config: Dict[str, Any] = config or {}
        super().__init__(name=name, config=base_config, metadata=metadata)
        # Every forcing input and the scheme-owned age state are required; a missing value is
        # an error, never a state default. shortwave_down (the surface downwelling shortwave flux,
        # CF surface_downwelling_shortwave_flux_in_air) feeds only the
        # radiation port diagnostic.
        self.optional_input_variables = {"shortwave_down"}
        
        # Set PhysicalField attributes  
        self.field_type = "surface_property"
        self.spatial_dimensions = 0
        self.field_units = "dimensionless"
        
        # Physical constants and shared parameters
        self.seconds_per_day = get_physical_constant("seconds_per_day")
        self.epsilon_small = float(
            self.get_parameter_value(
                "epsilon_small",
                namespace="hydrosphere.numerical", required=True)
        )
        
        # Hydrosphere snow parameters (YAML-first, required)
        self.albedo_scheme = str(
            self.get_parameter_value(
                "albedo_scheme",
                namespace="hydrosphere.snow", required=True)
        ).lower()
        if self.albedo_scheme not in {"bats", "age_based"}:
            raise ValueError(
                f"Unsupported snow albedo scheme '{self.albedo_scheme}'. "
                "Expected 'bats' or 'age_based'."
            )
        # The scheme-owned age state is read every call (its initial value is part of the
        # initial state, not a Functor default).
        if self.age_state_key not in self.metadata.input_variables:
            self.metadata.input_variables.append(self.age_state_key)
        self.fresh_snow_albedo = float(
            self.get_parameter_value(
                "fresh_snow_albedo",
                namespace="hydrosphere.snow", required=True)
        )
        self.old_snow_albedo = float(
            self.get_parameter_value(
                "old_snow_albedo",
                namespace="hydrosphere.snow", required=True)
        )
        self.bare_ground_albedo_factor = float(
            self.get_parameter_value(
                "bare_ground_albedo_factor",
                namespace="hydrosphere.snow", required=True)
        )
        self.snow_reset_threshold = float(
            self.get_parameter_value(
                "snow_reset_threshold",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_b_parameter = float(
            self.get_parameter_value(
                "bats_b_parameter",
                namespace="hydrosphere.snow", required=True)
        )
        self.dust_soot_factor = float(
            self.get_parameter_value(
                "dust_soot_factor",
                namespace="hydrosphere.snow", required=True)
        )
        self.albedo_decay_rate_cold = float(
            self.get_parameter_value(
                "albedo_decay_rate_cold",
                namespace="hydrosphere.snow", required=True)
        )
        self.albedo_decay_rate_warm = float(
            self.get_parameter_value(
                "albedo_decay_rate_warm",
                namespace="hydrosphere.snow", required=True)
        )
        self.temperature_threshold_warm = float(
            self.get_parameter_value(
                "temperature_threshold_warm",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_grain_growth_temperature_factor = float(
            self.get_parameter_value(
                "bats_grain_growth_temperature_factor",
                namespace="hydrosphere.snow", required=True)
        )
        self.age_based_temperature_sensitivity = float(
            self.get_parameter_value(
                "age_based_temperature_sensitivity",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_reference_temperature = float(
            self.get_parameter_value(
                "bats_reference_temperature",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_temperature_epsilon = float(
            self.get_parameter_value(
                "bats_temperature_epsilon",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_cos_zenith_threshold = float(
            self.get_parameter_value(
                "bats_cos_zenith_threshold",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_albedo_vis_base = float(
            self.get_parameter_value(
                "bats_albedo_vis_base",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_albedo_vis_age_factor = float(
            self.get_parameter_value(
                "bats_albedo_vis_age_factor",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_albedo_nir_base = float(
            self.get_parameter_value(
                "bats_albedo_nir_base",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_albedo_nir_age_factor = float(
            self.get_parameter_value(
                "bats_albedo_nir_age_factor",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_albedo_zenith_factor = float(
            self.get_parameter_value(
                "bats_albedo_zenith_factor",
                namespace="hydrosphere.snow", required=True)
        )
        self.bats_albedo_average_weight = float(
            self.get_parameter_value(
                "bats_albedo_average_weight",
                namespace="hydrosphere.snow", required=True)
        )
        self.aging_time_scale = float(
            self.get_parameter_value(
                "aging_time_scale",
                namespace="hydrosphere.snow", required=True)
        )
        self.minimum_snow_depth = float(
            self.get_parameter_value(
                "minimum_snow_depth",
                namespace="hydrosphere.snow", required=True)
        )
        self.albedo_clamp_min = float(
            self.get_parameter_value(
                "albedo_clamp_min",
                namespace="hydrosphere.snow", required=True)
        )
        self.albedo_clamp_max = float(
            self.get_parameter_value(
                "albedo_clamp_max",
                namespace="hydrosphere.snow", required=True)
        )
        # Reference albedo of the static absorption Hodge contract only; the runtime
        # albedo is the forward result and no parameter stands in for it.
        self.hodge_reference_surface_albedo = float(
            self.get_parameter_value(
                "hodge_reference_surface_albedo",
                namespace="hydrosphere.snow", required=True)
        )
        
        self.temperature_clamp_min = float(
            self.get_parameter_value(
                "temperature_clamp_min",
                namespace="hydrosphere.snow", required=True)
        )
        self.temperature_clamp_max = float(
            self.get_parameter_value(
                "temperature_clamp_max",
                namespace="hydrosphere.snow", required=True)
        )
        # Snow age state (initialized to zero)
        # Snow surface age is tracked in state, not as internal mutable state.

    def get_dependencies(self) -> DependencyDeclaration:
        """
        Return list of dependencies for this functor.
        
        Returns
        -------
        DependencyDeclaration
            Dependencies for snow process ordering
        """
        # Use lagged dependency to avoid cyclic ordering with snow mass balance.
        return self._merge_declared_dependencies({"required": [], "optional": [], "lagged": ["snow_pack"]})
    
    def get_provides(self) -> List[str]:
        """
        Return list of variables provided by this functor.
        
        Returns
        -------
        List[str]
            List of provided output variables
        """
        return [
            "surface_albedo",
            "snow_surface_age",
            "snow_nondimensional_age",
            "diagnostics.snow_albedo",
        ]

    @property
    def age_state_key(self) -> str:
        """Return the scheme-owned snow age state variable.

        References:
          - Dickinson et al. (1993), BATS nondimensional snow age tau_s.
          - Warren and Wiscombe (1980), age-based albedo decay.
        Mathematical expression:
          bats: tau_s with F_age = tau_s / (1 + tau_s); age_based: t_age in days.
        Parameter meanings:
          the configured hydrosphere.snow.albedo_scheme selects which state the
          Functor reads and writes; the other age state is left untouched.
        Units:
          snow_nondimensional_age [dimensionless]; snow_surface_age [day].
        Domain/constraints:
          both states are nonnegative; a scheme change is not an exact
          restart of the other scheme's state and is not claimed to be.
        Sign convention:
          aging is positive; fresh snowfall resets the active state to zero.
        """
        return "snow_nondimensional_age" if self.albedo_scheme == "bats" else "snow_surface_age"

    @property
    def age_tendency_key(self) -> str:
        """Return the diagnostics key of the active age tendency ([s-1] for bats, [day s-1] otherwise)."""
        return "snow_nondimensional_age_tendency" if self.albedo_scheme == "bats" else "snow_age_tendency"

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
        """Declare canonical cochain spaces for snow albedo observation.

        References:
          - Dickinson et al. (1993), Warren (1982), and Hirani (2003).
        Mathematical expression:
          alpha_s in C0(K), age in C0(K), a_sw = 1 - alpha_s.
        Parameter meanings:
          alpha_s is surface albedo [dimensionless], snow_surface_age is the
          age-based scheme's age [day], snow_nondimensional_age is the BATS
          nondimensional age tau_s [dimensionless], and a_sw is the absorbed
          shortwave fraction [dimensionless].
        """

        n_dofs = self._structural_n_dofs()
        return (
            CochainSpaceSpec(
                complex_id="hydrosphere_snow_albedo",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="dimensionless",
                extensivity="intensive",
                physical_domain="cryosphere_radiation",
                semantic_name="surface_albedo",
                orientation_convention="unsigned_fraction",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_snow_albedo",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="day",
                extensivity="intensive",
                physical_domain="cryosphere",
                semantic_name="snow_surface_age",
                orientation_convention="aging_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_snow_albedo",
                degree=0,
                primality="primal",
                n_dofs=n_dofs,
                units="dimensionless",
                extensivity="intensive",
                physical_domain="cryosphere",
                semantic_name="snow_nondimensional_age",
                orientation_convention="aging_positive",
                boundary_role="interior",
            ),
            CochainSpaceSpec(
                complex_id="hydrosphere_snow_albedo",
                degree=0,
                primality="dual",
                n_dofs=n_dofs,
                units="dimensionless",
                extensivity="intensive",
                physical_domain="surface_energy_balance",
                semantic_name="absorbed_shortwave_fraction",
                orientation_convention="into_surface_positive",
                boundary_role="port",
            ),
        )

    def required_parameters(self) -> tuple[str, ...]:
        """Declare ParameterManager keys used by runtime and structural builders."""

        return (
            "hydrosphere.numerical.epsilon_small",
            "hydrosphere.snow.albedo_scheme",
            "hydrosphere.snow.fresh_snow_albedo",
            "hydrosphere.snow.old_snow_albedo",
            "hydrosphere.snow.bare_ground_albedo_factor",
            "hydrosphere.snow.snow_reset_threshold",
            "hydrosphere.snow.bats_b_parameter",
            "hydrosphere.snow.dust_soot_factor",
            "hydrosphere.snow.albedo_decay_rate_cold",
            "hydrosphere.snow.albedo_decay_rate_warm",
            "hydrosphere.snow.temperature_threshold_warm",
            "hydrosphere.snow.bats_grain_growth_temperature_factor",
            "hydrosphere.snow.age_based_temperature_sensitivity",
            "hydrosphere.snow.bats_reference_temperature",
            "hydrosphere.snow.bats_temperature_epsilon",
            "hydrosphere.snow.bats_cos_zenith_threshold",
            "hydrosphere.snow.bats_albedo_vis_base",
            "hydrosphere.snow.bats_albedo_vis_age_factor",
            "hydrosphere.snow.bats_albedo_nir_base",
            "hydrosphere.snow.bats_albedo_nir_age_factor",
            "hydrosphere.snow.bats_albedo_zenith_factor",
            "hydrosphere.snow.bats_albedo_average_weight",
            "hydrosphere.snow.aging_time_scale",
            "hydrosphere.snow.minimum_snow_depth",
            "hydrosphere.snow.albedo_clamp_min",
            "hydrosphere.snow.albedo_clamp_max",
            "hydrosphere.snow.hodge_reference_surface_albedo",
            "hydrosphere.snow.temperature_clamp_min",
            "hydrosphere.snow.temperature_clamp_max",
        )

    def build_hodge_contract(
        self,
        n_dofs: Optional[int] = None,
        *,
        strict: bool = True,
    ) -> HodgeStarSpec:
        """Build the passive snow albedo radiation Hodge contract.

        References:
          - Warren (1982), Wiscombe and Warren (1980), and van der Schaft and Maschke (1995).
        Mathematical expression:
          star_alpha = diag(max(1 - alpha_ref, epsilon)).
        Parameter meanings:
          alpha_ref is hydrosphere.snow.hodge_reference_surface_albedo, the aged-snow
          reference albedo of this static contract (never a runtime albedo), and
          epsilon is hydrosphere.numerical.epsilon_small.
        Units:
          alpha_ref, epsilon and star_alpha are dimensionless.
        Domain/constraints:
          alpha_ref lies in its YAML bounds (0.1, 0.9); the floor epsilon keeps
          star_alpha positive definite.
        Sign convention:
          star_alpha maps the albedo observation to the absorbed shortwave fraction,
          positive into the surface.
        """

        n_dofs = self._structural_n_dofs(n_dofs=n_dofs)
        absorption = max(
            1.0 - float(self.hodge_reference_surface_albedo), float(self.epsilon_small)
        )
        values = torch.full(
            (n_dofs,),
            absorption,
            dtype=torch.float32,
            device=getattr(self, "device", torch.device("cpu")),
        )
        domain = CochainSpaceSpec(
            complex_id="hydrosphere_snow_albedo",
            degree=0,
            primality="primal",
            n_dofs=n_dofs,
            units="dimensionless",
            extensivity="intensive",
            physical_domain="cryosphere_radiation",
            semantic_name="surface_albedo_observation",
            orientation_convention="unsigned_fraction",
        )
        codomain = CochainSpaceSpec(
            complex_id="hydrosphere_snow_albedo",
            degree=0,
            primality="dual",
            n_dofs=n_dofs,
            units="dimensionless",
            extensivity="intensive",
            physical_domain="surface_energy_balance",
            semantic_name="absorbed_shortwave_fraction",
            orientation_convention="into_surface_positive",
            boundary_role="port",
        )
        hodge = HodgeStarSpec(
            name=f"{self.name}_snow_albedo_radiation_hodge",
            domain=domain,
            codomain=codomain,
            matrix=torch.diag(values),
            parameter_keys=(
                "hydrosphere.snow.hodge_reference_surface_albedo",
                "hydrosphere.numerical.epsilon_small",
            ),
            metric_keys=("hydrosphere.snow.minimum_snow_depth",),
            material_law="shortwave_absorption_fraction",
        )
        if strict:
            hodge.validate(passive=True)
        return hodge

    def build_hodge_terms(
        self,
        complex_bundle: Any = None,
        resolved_parameters: Optional[Mapping[str, Any]] = None,
    ) -> tuple[HodgeStarSpec, ...]:
        """Build snow albedo Hodge terms for IPHS assembly."""

        return (
            self.build_hodge_contract(
                n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
                strict=True,
            ),
        )

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the shortwave partition terms for structural audits."""

        return {
            "reflected_shortwave_fraction": "alpha_s",
            "absorbed_shortwave_fraction": "1 - alpha_s",
            "radiation_port_power": "P_rad = I_sw * (1 - alpha_s)",
        }

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return nonnegative radiative dissipation statement."""

        return {
            "radiative_diagnostic_entropy": "sigma = I_sw * (1 - alpha_s) / T_surface >= 0",
            "snow_age_dissipation": "sigma_age = (d age_s / dt)^2 >= 0",
        }

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> None:
        """Return no reversible interconnection for diagnostic albedo."""

        return None

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any = None,
    ) -> Mapping[str, Any]:
        """Return the passive radiation absorption mobility.

        References:
          - Warren (1982), Wiscombe and Warren (1980), and van der Schaft and Maschke (1995).
        Mathematical expression:
          M_h = star_alpha, sigma = I_sw.T M_h I_sw / T >= 0.
        Parameter meanings:
          star_alpha maps incoming shortwave radiation to absorbed fraction.
        """

        hodge = self.build_hodge_contract(
            n_dofs=self._structural_n_dofs(complex_bundle=complex_bundle),
            strict=True,
        )
        matrix = hodge.matrix
        return {
            "state_vars": tuple(
                f"snow_albedo_observation[{idx}]" for idx in range(int(matrix.shape[0]))
            ),
            "J": torch.zeros_like(matrix),
            "M": matrix,
            "ports": self.ports(),
        }

    def ports(self) -> tuple[PortSpecIPHS, ...]:
        """Declare the radiation port for snow albedo partitioning.

        References:
          - Warren (1982) and van der Schaft and Maschke (1995).
        Mathematical expression:
          P_rad = I_sw * (1 - alpha_s), with positive power into the surface.
        Parameter meanings:
          I_sw is incoming shortwave radiation [W m-2] and alpha_s is albedo.
        """

        effort_space = CochainSpaceSpec(
            complex_id="hydrosphere_snow_albedo",
            degree=0,
            primality="primal",
            n_dofs=1,
            units="W m-2",
            extensivity="intensive",
            physical_domain="surface_energy_balance",
            semantic_name="shortwave_down",
            orientation_convention="downward_positive",
            boundary_role="port",
        )
        flow_space = CochainSpaceSpec(
            complex_id="hydrosphere_snow_albedo",
            degree=0,
            primality="dual",
            n_dofs=1,
            units="dimensionless",
            extensivity="intensive",
            physical_domain="cryosphere_radiation",
            semantic_name="absorbed_shortwave_fraction",
            orientation_convention="into_surface_positive",
            boundary_role="port",
        )
        return (
            PortSpecIPHS(
                name="snow_albedo_radiation_port",
                effort_var="shortwave_down",
                flow_var="snow_absorbed_shortwave_fraction",
                effort_space=effort_space,
                flow_space=flow_space,
                power_units="W m-2",
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
    
    def check_conservation_laws(
        self,
        input_state: StateDict,
        output_updates: FluxDict,
    ) -> Dict[str, float]:
        """Return the shortwave partition closure residual for the declared energy law.

        References:
          - Warren (1982) Rev. Geophys. 20:67-89, snow albedo as reflected fraction.
          - RELIC certificate hydrosphere_snow_albedo_snow_albedo.yaml, conservation_map.
        Mathematical expression:
          r_E = max_i |alpha_s,i + a_sw,i - 1|, with a_sw = 1 - alpha_s the absorbed
          fraction; the certificate balance I_abs + I_ref = I_sw follows for any I_sw.
        Parameter meanings:
          alpha_s is the emitted surface albedo, a_sw the routed absorbed shortwave
          fraction diagnostic, i the grid cell.
        Units:
          alpha_s, a_sw, r_E [dimensionless].
        Domain/constraints:
          both outputs must be present; albedo and age are intensive REPLACE states,
          so their sum is not a conserved quantity and is not used.
        Sign convention:
          r_E >= 0; a positive value means reflected plus absorbed fractions do not
          partition the incident shortwave flux.
        """
        if "surface_albedo" not in output_updates or "diagnostics.snow_albedo" not in output_updates:
            raise ValueError(
                f"{self.name}: energy partition check requires surface_albedo and "
                "diagnostics.snow_albedo outputs"
            )
        albedo = output_updates["surface_albedo"]
        if isinstance(albedo, UpdateItem):
            albedo = albedo.value
        absorbed = output_updates["diagnostics.snow_albedo"]["snow_absorbed_shortwave_fraction"]
        albedo = self._ensure_tensor(albedo)
        absorbed = self._ensure_tensor(absorbed).to(device=albedo.device, dtype=albedo.dtype)
        residual = torch.abs(albedo + absorbed - 1.0)
        return {"energy": float(residual.detach().max().item())}

    def forward(self,
        # References:
        #   Dickinson et al. (1993), BATS Version 1e; Warren and Wiscombe (1980); Warren (1982).
        # Mathematical expression:
        #   a_s <- 0 on fresh snowfall above the reset threshold;
        #   bats: tau_s += dt k_decay(T) (r1 + r2 + r3), F_age = tau_s / (1 + tau_s);
        #   age_based: t_age += dt / 86400, alpha = alpha_old + (alpha_fresh - alpha_old)
        #   exp(-t_age f_T / tau); alpha = clip(alpha_raw, alpha_min, alpha_max);
        #   a_sw = 1 - alpha; P_sw = I_sw a_sw.
        # Parameter meanings:
        #   a_s is the scheme-owned age state (snow_nondimensional_age for bats,
        #   snow_surface_age for age_based); alpha the reflected fraction; I_sw the
        #   optional incoming shortwave; snow_present selects snow versus the configured
        #   bare-ground albedo.
        # Units:
        #   tau_s, alpha, a_sw, F_age [1]; t_age, tau [day]; k_decay [s-1];
        #   I_sw, P_sw [W m-2]; depth [m]; temperature [K].
        # Domain/constraints:
        #   Only the active scheme's age state is read and written; the clipping
        #   residual and the partition closure |alpha + a_sw - 1| are routed as
        #   diagnostics; the optional shortwave diagnostic does not advance snow mass.
        # Sign convention:
        #   Aging lowers alpha; positive P_sw enters the surface.
                state: StateDict, 
                time: float, 
                dt: float,
                **kwargs) -> FluxDict:
        """
        Compute dynamic snow surface albedo.
        
        Parameters
        ----------
        state : StateDict
            Current system state including snow and meteorological variables
        time : float
            Current simulation time [s]
        dt : float
            Time step [s]
        **kwargs
            Additional context variables
            
        Returns
        -------
        FluxDict
            Surface albedo and diagnostic variables
            
        Raises
        ------
        ValueError
            If required state variables are missing
        """
        missing = [name for name in self.get_required_inputs() if state.get(name) is None]
        if missing:
            raise ValueError(
                "SnowAlbedo requires the inputs " + ", ".join(missing)
                + "; no state default is substituted"
            )
        state = {
            **state,
            **{name: self._ensure_tensor(state[name]) for name in self.get_required_inputs()},
        }

        snowfall = state["snowfall"]  # kg m-2 s-1
        snow_surface_temp = state["snow_surface_temperature"]  # K
        air_temp = state["air_temperature"]  # K
        solar_zenith = state["solar_zenith_angle"]  # radians
        snow_depth = state["snow_depth"]  # m

        # References: Dickinson et al. (1993), BATS Version 1e (snow aging advances with the
        #   step length); Molotch and Bales (2006); Warren and Wiscombe (1980).
        # Mathematical expression: dt_days = dt / seconds_per_day; the age tendency of the
        #   step is (a_s^{n+1} - a_s^n) / dt.
        # Parameter meanings: dt the step length; seconds_per_day the exact conversion of the
        #   physical-constant SSOT; a_s the scheme-owned age state.
        # Units: dt [s]; seconds_per_day [s day-1]; dt_days [day].
        # Domain/constraints: dt must be finite and positive because the age tendency divides
        #   by it; a non-positive dt raises instead of being clamped by the dimensionless
        #   hydrosphere.numerical.epsilon_small, and the exact constant is not clamped.
        # Sign convention: dt is a positive duration; aging increases the active age state.
        dt_tensor = self._ensure_tensor(dt).to(device=snow_depth.device, dtype=snow_depth.dtype)
        if not bool(torch.all(torch.isfinite(dt_tensor) & (dt_tensor > 0.0))):
            raise ValueError(
                f"SnowAlbedo requires a finite positive step length dt, got {dt}"
            )
        seconds_per_day = self._ensure_tensor(self.seconds_per_day).to(
            device=snow_depth.device, dtype=snow_depth.dtype
        )
        dt_seconds = dt_tensor
        dt_days = dt_tensor / seconds_per_day

        # References: Dickinson et al. (1993); Warren and Wiscombe (1980).
        # Mathematical expression: the active age state is a_s = tau_s for the
        #   BATS scheme (nondimensional, F_age = tau_s/(1+tau_s)) or a_s = t_age
        #   for the age-based scheme (days); the other state is left untouched.
        # Parameter meanings: age_state_key names the scheme-owned StateManager
        #   variable, a required input (its initial value belongs to the initial state).
        # Units: snow_nondimensional_age [1]; snow_surface_age [day].
        # Domain/constraints: nonnegative; fresh snowfall above the reset threshold
        #   zeroes the active state before aging is applied.
        # Sign convention: aging increases the active state.
        age_key = self.age_state_key
        snow_age_state = self._ensure_tensor(state[age_key]).to(
            device=snow_depth.device, dtype=snow_depth.dtype
        )

        # Check if snow is present (minimum depth threshold)
        snow_present = snow_depth > self.minimum_snow_depth

        # 1. Check for fresh snowfall reset
        fresh_snow_reset = snowfall > self.snow_reset_threshold
        snow_age_state = torch.where(
            fresh_snow_reset, torch.zeros_like(snow_age_state), snow_age_state
        )
        age_state_start = snow_age_state

        # 2. Compute snow aging based on selected scheme
        if self.albedo_scheme == "bats":
            surface_albedo, aging_rate, updated_age = self._compute_bats_albedo(
                snow_surface_temp, solar_zenith, dt_seconds, snow_age_state
            )
        else:  # age_based
            surface_albedo, aging_rate, updated_age = self._compute_age_based_albedo(
                air_temp, dt_days, snow_age_state
            )
        
        # References: Warren (1982); Wiscombe and Warren (1980); RELIC Phase 8 residual routing.
        # Mathematical expression: r_clip = max(alpha_min - alpha_raw, 0) + max(alpha_raw - alpha_max, 0).
        # Parameter meanings: alpha_min and alpha_max are YAML albedo bounds; alpha_raw is the computed snow albedo.
        # 3. Ensure albedo is within physical bounds and route projection residuals.
        surface_albedo = self._ensure_tensor(surface_albedo)
        raw_surface_albedo = surface_albedo
        albedo_clipping_residual = (
            torch.clamp(self.albedo_clamp_min - raw_surface_albedo, min=0.0)
            + torch.clamp(raw_surface_albedo - self.albedo_clamp_max, min=0.0)
        )
        surface_albedo = torch.clamp(
            raw_surface_albedo,
            min=self.albedo_clamp_min,
            max=self.albedo_clamp_max
        )

        # Apply snow-present mask to outputs
        bare_albedo = torch.full_like(
            surface_albedo,
            self.old_snow_albedo * self.bare_ground_albedo_factor,
        )
        surface_albedo = torch.where(snow_present, surface_albedo, bare_albedo)
        updated_age = torch.where(snow_present, updated_age, torch.zeros_like(updated_age))
        aging_rate = torch.where(snow_present, aging_rate, torch.zeros_like(aging_rate))
        albedo_clipping_residual = torch.where(
            snow_present,
            albedo_clipping_residual,
            torch.zeros_like(albedo_clipping_residual),
        )
        snow_absorbed_shortwave_fraction = torch.clamp(
            1.0 - surface_albedo,
            min=0.0,
            max=1.0,
        )
        # References: Warren (1982); van der Schaft and Maschke (1995); RELIC BIHSS Gate 0.
        # Mathematical expression: P_rad = I_sw * (1 - alpha_s) when I_sw is supplied;
        #   d a_s/dt = (a_s^{n+1} - a_s^n) / dt.
        # Parameter meanings: I_sw is the optional shortwave_down input (surface downwelling
        #   shortwave flux incident on the snow surface, CF surface_downwelling_shortwave_flux_in_air; the
        #   port effort), alpha_s the surface albedo, a_s the scheme-owned age state: tau_s
        #   for bats or t_age for age_based.
        # Units: I_sw and P_rad [W m-2]; alpha_s [1]; tau_s [1] with tendency [s-1]; t_age
        #   [day] with tendency [day s-1]; dt [s].
        # Domain/constraints: without I_sw there is no port power, so the diagnostic is
        #   omitted (an explicit optional branch; no zero is substituted); dt > 0 was
        #   checked above.
        # Sign convention: P_rad >= 0 is absorbed into the surface; aging makes the
        #   tendency positive.
        incoming_shortwave = state.get("shortwave_down")
        snow_albedo_radiation_port_power = None
        if incoming_shortwave is not None:
            snow_albedo_radiation_port_power = (
                self._ensure_tensor(incoming_shortwave).to(
                    device=snow_absorbed_shortwave_fraction.device,
                    dtype=snow_absorbed_shortwave_fraction.dtype,
                )
                * snow_absorbed_shortwave_fraction
            )
        snow_age_tendency = (updated_age - age_state_start) / dt_seconds.to(
            device=updated_age.device, dtype=updated_age.dtype
        )
        snow_age_tendency = torch.where(
            snow_present,
            snow_age_tendency,
            torch.zeros_like(snow_age_tendency),
        )

        diagnostics = {
            "aging_rate": aging_rate,
            "fresh_snow_reset": fresh_snow_reset,
            "snow_present": snow_present,
            "albedo_clipping_residual": albedo_clipping_residual,
            "snow_absorbed_shortwave_fraction": snow_absorbed_shortwave_fraction,
            self.age_tendency_key: snow_age_tendency,
        }
        if snow_albedo_radiation_port_power is not None:
            diagnostics["snow_albedo_radiation_port_power"] = snow_albedo_radiation_port_power

        result = {
            "surface_albedo": self.wrap_update(surface_albedo, mode=UpdateMode.REPLACE),
            "diagnostics.snow_albedo": diagnostics,
        }
        # Both age states are declared outputs; only the scheme-owned one is written.
        # Literal keys keep the static output-contract scan resolvable (age_key is
        # the same selection used for the state read above).
        age_update = self.wrap_update(updated_age, mode=UpdateMode.REPLACE)
        if age_key == "snow_nondimensional_age":
            result["snow_nondimensional_age"] = age_update
        else:
            result["snow_surface_age"] = age_update
        return result
    
    # References: Dickinson et al. (1993), BATS Version 1e; Molotch and Bales (2006); RHESSys snowpack_daily_F.c.
    # Mathematical expression: tau_s^{n+1} = tau_s^n + dt * k_decay(T) * (r1 + r2 + r3);
    #   r1 = exp(b (1/T_ref - 1/(T + eps))), r2 = min(r1^10, 1), r3 = c_soot;
    #   F_age = tau_s/(1 + tau_s); alpha = w*alpha_vis + (1-w)*alpha_nir.
    # Parameter meanings: tau_s is the BATS nondimensional snow age (BATS uses 1/tau_0
    #   with tau_0 = 1e6 s; RELIC uses the cold/warm decay rates k_decay in that role);
    #   r1 grain growth by vapor diffusion with b = bats_grain_growth_temperature_factor
    #   (5000 K, BATS 1e; RHESSys snowpack_daily_F.c: exp(5000*(1/273.16 - 1/T))),
    #   T_ref = bats_reference_temperature, eps = bats_temperature_epsilon;
    #   r2 melt-freeze, r3 = dust_soot_factor dust and soot.
    # Units: tau_s, F_age, r1, r2, r3 [1]; k_decay [s-1]; dt [s]; b, T_ref, eps, T [K];
    #   solar zenith [rad].
    # Domain/constraints: tau_s >= 0; T is clamped to the configured BATS range;
    #   tau_s is not an age in days and is stored in snow_nondimensional_age; dt > 0
    #   (checked by forward), so the aging rate divides by dt without a clamp.
    # Sign convention: aging increases tau_s and lowers albedo.
    def _compute_bats_albedo(self,
                           surface_temp: Tensor,
                           solar_zenith: Tensor,
                           dt_seconds: Tensor,
                           snow_surface_age: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute snow albedo using BATS parameterization.
        
        Based on RHESSys implementation following Dickinson et al. (1993)
        and Molotch & Bales (2006).
        
        Mathematical Expression
        -----------------------
        r1 = exp(grain_factor * (1 / T_ref - 1 / (T_surface + temp_epsilon)))
        r2 = min(r1^10, 1.0)
        r3 = dust_soot_factor
        tau_s += dt_seconds * decay_rate * (r1 + r2 + r3)
        fage = tau_s / (1 + tau_s)
        albedo = w * albedo_vis + (1 - w) * albedo_nir

        Parameter meanings
        ------------------
        surface_temp: snow surface temperature [K]
        solar_zenith: solar zenith angle [rad]
        dt_seconds: time step [s]
        snow_surface_age: current BATS nondimensional snow age tau_s [dimensionless]
        """
        # References: Dickinson et al. (1993); Molotch and Bales (2006); RHESSys snowpack_daily_F.c
        # Mathematical expression: tau_s += dt_seconds * decay_rate * (r1 + r2 + r3)
        # Parameter meanings: decay_rate [s-1]; dt_seconds [s]; tau_s, r1/r2/r3 [1]
        # Temperature effect on aging (vapor diffusion)
        # Note: BATS formula requires absolute temperature in Kelvin throughout
        # Clamp surface temperature to physical bounds (223.15K to 283.15K = -50degC to +10degC)
        surface_temp_clamped = torch.clamp(
            surface_temp,
            min=self.temperature_clamp_min,
            max=self.temperature_clamp_max
        )
        
        # BATS aging rate formula - uses Kelvin temperatures directly
        # r1 = exp(grain_factor * (1 / T_ref - 1 / (T_surface + temp_epsilon))) with
        # grain_factor = 5000 K (BATS 1e; RHESSys snowpack_daily_F.c). The former
        # dimensionless grain_growth_coefficient (0.01) scaled the exponent to 50 K and
        # weakened the temperature sensitivity a hundredfold; it has no source in BATS.
        grain_factor = self._ensure_tensor(self.bats_grain_growth_temperature_factor).to(
            device=surface_temp_clamped.device, dtype=surface_temp_clamped.dtype
        )
        reference_temp = self._ensure_tensor(self.bats_reference_temperature).to(
            device=surface_temp_clamped.device, dtype=surface_temp_clamped.dtype
        )
        temp_epsilon = self._ensure_tensor(self.bats_temperature_epsilon).to(
            device=surface_temp_clamped.device, dtype=surface_temp_clamped.dtype
        )
        r1 = torch.exp(
            grain_factor * (1.0 / reference_temp - 1.0 / (surface_temp_clamped + temp_epsilon))
        )
        
        # Meltwater refreeze effect
        r2 = torch.clamp(r1**10, max=1.0)
        
        # Dust and soot contamination effect
        r3 = self._ensure_tensor(self.dust_soot_factor).to(
            device=surface_temp_clamped.device, dtype=surface_temp_clamped.dtype
        )
        
        # Update snow surface age
        decay_rate_cold = self._ensure_tensor(self.albedo_decay_rate_cold).to(
            device=surface_temp_clamped.device, dtype=surface_temp_clamped.dtype
        )
        decay_rate_warm = self._ensure_tensor(self.albedo_decay_rate_warm).to(
            device=surface_temp_clamped.device, dtype=surface_temp_clamped.dtype
        )
        temp_threshold = self._ensure_tensor(self.temperature_threshold_warm).to(
            device=surface_temp_clamped.device, dtype=surface_temp_clamped.dtype
        )
        decay_rate = torch.where(
            surface_temp_clamped >= temp_threshold, decay_rate_warm, decay_rate_cold
        )
        aging_increment = dt_seconds * decay_rate * (r1 + r2 + r3)
        updated_age = snow_surface_age + aging_increment
        aging_rate = aging_increment / dt_seconds
        
        # Normalized aging factor
        fage = updated_age / (1.0 + updated_age)
        
        # Solar zenith angle effect
        cos_zenith = torch.cos(solar_zenith)
        bats_b = self._ensure_tensor(self.bats_b_parameter).to(
            device=cos_zenith.device, dtype=cos_zenith.dtype
        )
        cos_threshold = self._ensure_tensor(self.bats_cos_zenith_threshold).to(
            device=cos_zenith.device, dtype=cos_zenith.dtype
        )
        fz = torch.where(
            cos_zenith < cos_threshold,
            (1.0 / bats_b) * (
                (1.0 + bats_b) / (1.0 + 2.0 * bats_b * cos_zenith) - 1.0
            ),
            torch.zeros_like(cos_zenith),
        )
        
        # Visible and near-infrared albedo components
        albedo_vis_base = self._ensure_tensor(self.bats_albedo_vis_base).to(
            device=fage.device, dtype=fage.dtype
        )
        albedo_vis_age_factor = self._ensure_tensor(self.bats_albedo_vis_age_factor).to(
            device=fage.device, dtype=fage.dtype
        )
        albedo_nir_base = self._ensure_tensor(self.bats_albedo_nir_base).to(
            device=fage.device, dtype=fage.dtype
        )
        albedo_nir_age_factor = self._ensure_tensor(self.bats_albedo_nir_age_factor).to(
            device=fage.device, dtype=fage.dtype
        )
        zenith_factor = self._ensure_tensor(self.bats_albedo_zenith_factor).to(
            device=fage.device, dtype=fage.dtype
        )
        albedo_vis = (
            albedo_vis_base * (1.0 - albedo_vis_age_factor * fage)
            + zenith_factor * fz * (1.0 - albedo_vis_base * (1.0 - albedo_vis_age_factor * fage))
        )
        albedo_nir = (
            albedo_nir_base * (1.0 - albedo_nir_age_factor * fage)
            + zenith_factor * fz * (1.0 - albedo_nir_base * (1.0 - albedo_nir_age_factor * fage))
        )
        
        # Spectrally-averaged surface albedo
        albedo_weight = self._ensure_tensor(self.bats_albedo_average_weight).to(
            device=fage.device, dtype=fage.dtype
        )
        surface_albedo = albedo_weight * albedo_vis + (1.0 - albedo_weight) * albedo_nir
        
        return surface_albedo, aging_rate, updated_age
    
    # References: Warren and Wiscombe (1980); Dickinson et al. (1993).
    # Mathematical expression: t_age^{n+1} = t_age^n + dt_days;
    #   f_T = exp(kappa*(T - T_warm)) for T > T_warm else 1;
    #   alpha = alpha_old + (alpha_fresh - alpha_old) * exp(-t_age * f_T / tau).
    # Parameter meanings: t_age snow age in days stored in snow_surface_age; kappa the
    #   age-based warm-aging sensitivity; T_warm the warm threshold; tau the aging scale.
    # Units: t_age, dt_days, tau [day]; kappa [K-1]; T, T_warm [K]; alpha [1].
    # Domain/constraints: t_age >= 0; kappa multiplies a temperature difference, so it
    #   must carry K-1 and is not the dimensionless BATS grain growth coefficient.
    # Sign convention: aging and warmth lower the albedo.
    def _compute_age_based_albedo(self,
                                air_temp: Tensor,
                                dt_days: Tensor,
                                snow_surface_age: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Compute snow albedo using simple age-based exponential decay.
        
        Mathematical Expression
        -----------------------
        surface_age += dt_days
        albedo = old_albedo + (fresh_albedo - old_albedo) * exp(-age/tau)
        
        where tau is the characteristic aging time scale.
        """
        # References: Dickinson et al. (1993); Warren and Wiscombe (1980).
        # Mathematical expression: t_age^{n+1} = t_age^n + dt_days; aging rate =
        #   dt_days / dt_days; albedo = old + (fresh - old) * exp(-t_age f_T / tau).
        # Parameter meanings: air_temp the air temperature; dt_days the step in days;
        #   snow_surface_age t_age, the scheme-owned age state; tau the aging time scale;
        #   f_T the temperature factor applied below.
        # Units: air_temp [K]; dt_days, t_age and tau [day]; aging rate [day day-1];
        #   albedo [1].
        # Domain/constraints: dt_days > 0 because forward checks dt > 0, so the rate
        #   divides by the step itself; no dimensionless epsilon bounds a duration.
        # Sign convention: the age increases with time and the albedo decays toward the
        #   old-snow value.
        dt_days = dt_days.to(device=air_temp.device, dtype=air_temp.dtype)
        aging_increment = dt_days
        updated_age = snow_surface_age + aging_increment
        aging_rate = aging_increment / dt_days
        
        # Temperature-dependent aging rate adjustment
        # Use Kelvin temperature directly, convert to relative scale for exponential
        temp_threshold = self._ensure_tensor(self.temperature_threshold_warm).to(
            device=air_temp.device, dtype=air_temp.dtype
        )
        temp_relative = air_temp - temp_threshold  # K
        warm_sensitivity = self._ensure_tensor(self.age_based_temperature_sensitivity).to(
            device=air_temp.device, dtype=air_temp.dtype
        )  # K-1
        temp_factor = torch.where(
            air_temp > temp_threshold,
            torch.exp(temp_relative * warm_sensitivity),
            torch.ones_like(air_temp)
        )
        adjusted_age = updated_age * temp_factor
        
        # Exponential decay from fresh to old snow albedo
        aging_scale = self._ensure_tensor(self.aging_time_scale).to(
            device=air_temp.device, dtype=air_temp.dtype
        )
        decay_factor = torch.exp(-adjusted_age / aging_scale)
        surface_albedo = (
            self.old_snow_albedo + 
            (self.fresh_snow_albedo - self.old_snow_albedo) * decay_factor
        )
        
        return surface_albedo, aging_rate, updated_age
    
    # PhysicalField interface implementation
    def get_value_at(self, coordinates: Tensor) -> Tensor:
        """
        No coordinate query: SnowAlbedo is a point model.

        Notes
        -----
        surface_albedo is written by this Functor and committed by StateManager
        (SPATIAL_DIMS = None). Returning a parameter would conceal a missing
        state and create a second albedo source; read the committed state.
        """
        raise NotImplementedError(
            "SnowAlbedo is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read the committed surface_albedo from StateManager"
        )

    def compute_gradient(self, coordinates: Tensor) -> Tensor:
        """
        No spatial gradient: SnowAlbedo is a point model.

        A point model has no coordinate axis, so a gradient of any shape would
        be fabricated.
        """
        raise NotImplementedError(
            "SnowAlbedo is a point model (SPATIAL_DIMS = None): it has no "
            "spatial gradient"
        )

    def interpolate(self, source_coordinates: Tensor, target_coordinates: Tensor) -> Tensor:
        """
        No interpolation: SnowAlbedo is a point model.

        Notes
        -----
        surface_albedo is committed by StateManager (SPATIAL_DIMS = None).
        """
        raise NotImplementedError(
            "SnowAlbedo is a point model (SPATIAL_DIMS = None): it has no "
            "interpolation; read the committed surface_albedo from StateManager"
        )
    
    def update_field(self, time_step: float, source_terms: Dict, boundary_conditions: Optional[Dict] = None) -> None:
        """
        Update albedo field (implemented via forward method).
        
        Parameters
        ----------
        time_step : float
            Time step [s]
        source_terms : Dict
            Source terms (snowfall, temperature, etc.)
        boundary_conditions : Optional[Dict]
            Boundary conditions (not used for albedo)
        """
        # References: FunctorBase update adapter for PhysicalField.
        # Mathematical expression: field_update = forward(state, time, dt).
        # Parameter meanings: time_step [s], source_terms [dict], boundary_conditions [dict].
        super().update_field(time_step, source_terms, boundary_conditions)
        return None
    
    def _ensure_tensor(self, value: Union[float, Tensor]) -> Tensor:
        """
        Ensure input is a tensor with proper dtype.

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


# Export the main class
__all__ = ["SnowAlbedo"]

