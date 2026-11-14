"""Big-leaf canopy energy balance giving the leaf temperature (linearised Penman-Monteith form).

The canopy is one big leaf per unit ground area that receives the available energy
A = R_n - G of the surface and exchanges sensible heat H and latent heat lambda E with
the air at the reference height through the aerodynamic resistance r_a (heat and water
vapour) and, for water vapour only, the canopy (bulk stomatal) resistance r_c = 1 / g_c.
Linearising the saturation vapour pressure about the air temperature,
e_s(T_l) = e_s(T_a) + s (T_l - T_a), closes the balance A = H + lambda E in closed form.

References
----------
- Monteith, J.L. (1965). Evaporation and environment. Symp. Soc. Exp. Biol. 19, 205-224
  (combination equation; the big-leaf surface whose latent heat this balance shares).
- Allen, R.G., Pereira, L.S., Raes, D. and Smith, M. (1998). Crop evapotranspiration.
  FAO Irrigation and Drainage Paper 56: Eq. 3 (Penman-Monteith), Eq. 4 (aerodynamic
  resistance), Eq. 42 (daily soil heat flux G = 0).
- Jones, H.G. (2014). Plants and Microclimate, 3rd ed. Cambridge University Press,
  Chapter 9 (leaf temperature from the linearised energy balance; Eq. 9.6 in the RELIC
  plan, equation number not checked against the book here).
- Campbell, G.S. and Norman, J.M. (1998). An Introduction to Environmental Biophysics,
  2nd ed. Springer, Chapter 14 (Eq. 14.6, the same linearisation in molar form).
- Monteith, J.L. and Unsworth, M.H. (2013). Principles of Environmental Physics, 4th
  ed. Academic Press, Chapters 13-14 (surface temperature of a big-leaf canopy).

Mathematical Expressions
------------------------
Resistance form (Jones 2014, Chapter 9) with r_HR = r_aW = r_a (no separate radiative
resistance, see Domain below):
    T_l - T_a = [r_a (r_a + r_c) gamma A / (rho c_p) - r_a D] / [gamma (r_a + r_c) + s r_a]
Implemented conductance form (multiply numerator and denominator by g_a g_c, g = 1/r):
    dT = [gamma (g_a + g_c) A / (rho c_p) - g_a g_c D] / (g_a [s g_c + gamma (g_a + g_c)])
    H  = rho c_p g_a dT
    lambda E = A - H = [s A + rho c_p D g_a] g_c / [s g_c + gamma (g_a + g_c)]
             (identical to the Penman-Monteith latent heat of FAO-56 Eq. 3)
Limits: g_c -> 0 gives dT = A / (rho c_p g_a) (dry leaf, H = A); g_c -> infinity gives
dT = [gamma A / (rho c_p g_a) - D] / (s + gamma), the wet-surface limit whose A = 0
value -D / (s + gamma) is the linearised wet-bulb depression.
Daily maximum (temporal_support "daily"): T_l,max = T_max + dT (declared shift).

Units
-----
T_a, T_l, T_max [K]; dT [K]; A, R_n, G, H, lambda E [W m-2]; D [Pa]; s, gamma [Pa K-1];
rho [kg m-3]; c_p [J kg-1 K-1]; g_a, g_c [m s-1]; r_a, r_c [s m-1].
"""

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from relic.core.abstractions import FluxDict, StateDict
from relic.core.functor import (
    BiologicalProcessFunctor,
    ConflictResolutionPolicy,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    TemporalScale,
    VariableKind,
)
from relic.core.state import UpdateItem, UpdateMode
from relic.scientific.hydrosphere.evapotranspiration.penman_monteith import (
    fao56_aerodynamic_conductance,
)
from relic.scientific.hydrosphere.utils.psychrometrics import (
    compute_moist_air_density,
    compute_psychrometric_constant,
    compute_saturation_vapor_pressure,
    compute_slope_saturation_vapor_pressure,
)
from relic.utils.physical_constants import get_physical_constant
from relic.utils.unified_response_functions import beer_lambert_effective_leaf_area, validate_temperature_range
from relic.scientific.atmosphere import daylight_quadrature
from relic.scientific.atmosphere.most.stability_parameters import friction_velocity, monin_obukhov_length, zeta as most_zeta
from relic.scientific.atmosphere.most.surface_fluxes import aerodynamic_resistance


class LeafEnergyBalance(BiologicalProcessFunctor):
    """Diagnostic big-leaf energy balance: leaf temperature from canopy conductance and radiation.

    Read-only diagnostic (no prognostic state): every call solves the steady linearised
    balance A = H + lambda E for the inputs of that call and REPLACEs leaf_temperature.
    """

    FUNCTOR_NAME = "leaf_energy_balance"
    # Declared temporal supports; config["temporal_support"] must name one of them, there
    # is no implicit default:
    #   "daily": FAO-56 24-hour support (Allen et al. 1998): every input is the 24-hour
    #     mean of its quantity, G = 0 (Eq. 42), dt = seconds_per_day only; also reads the
    #     daily maximum and minimum air temperature: e_s of the vapour pressure in the air
    #     density is the Eq. 12 mean (e_s(T_max) + e_s(T_min)) / 2, as in the daily
    #     PenmanMonteithET, and the declared daily-maximum estimate
    #     leaf_temperature_max = T_max + dT is written.
    #   "instantaneous": any dt; G is the declared signed input soil_heat_flux [W m-2]
    #     on the support of net_radiation; e_s = e_s(T_a); no daily maximum is written.
    #   "daylight_gauss": daily step only (dt = seconds_per_day); the instantaneous
    #     balance at the daylight Gauss samples of
    #     relic.scientific.atmosphere.daylight_quadrature, averaged with the sample
    #     weights: T_k the SWAT+ Eq. 1:1.3.1 course of the daily mean, maximum and
    #     minimum; D_k = max(e0(T_k) - e_a, 0) (actual_vapor_pressure replaces
    #     vapor_pressure_deficit); R_n,k = R_n - S_n + S_n,k with the daily net
    #     shortwave S_n redistributed by the solar elevation (net longwave constant over
    #     the day); g_c,k the per-sample Medlyn canopy conductance
    #     (canopy_conductance_daylight_samples replaces canopy_conductance); G_k = 0.
    #     Writes the 24-hour means, leaf_temperature_max = T_max + dT of the daylight
    #     instant with the highest air temperature, and leaf_temperature_daylight_samples.
    TEMPORAL_SUPPORTS = ("daily", "instantaneous", daylight_quadrature.DAYLIGHT_GAUSS)
    DAILY_INPUTS = ("air_temperature_max", "air_temperature_min")
    DAILY_OUTPUTS = ("leaf_temperature_max",)
    INSTANTANEOUS_INPUTS = ("soil_heat_flux",)
    DAYLIGHT_INPUTS = ("air_temperature_max", "air_temperature_min", "shortwave_net",
                       "latitude", "day_of_year")
    DAYLIGHT_OUTPUTS = ("leaf_temperature_max", "leaf_temperature_daylight_samples")
    DAYLIGHT_REPLACED_INPUTS = {"vapor_pressure_deficit": "actual_vapor_pressure",
                                "canopy_conductance": "canopy_conductance_daylight_samples"}
    DIAGNOSTIC_PREFIX = "diagnostics.leaf_energy_balance."
    # Declared forms of the balance; config["energy_balance_form"], and an absent key keeps
    # "linearised_big_leaf" (the form above, bitwise as commit d9a4637):
    #   "linearised_big_leaf": r_HR = r_a; all of A = R_n - G goes to the big leaf; g_a from
    #     hydrosphere.numerical.minimum_wind_speed.
    #   "radiative_canopy_share": (1) radiative coupling with the isothermal net radiation
    #     R_ni = S_n + eps L_down - eps sigma T_a^4 (the net radiation the surface would
    #     receive at air temperature) and the heat transfer conductance
    #     g_HR = g_a + g_R, g_R = 4 eps sigma T_a^3 / (rho c_p), the first-order Taylor
    #     term of eps sigma T_l^4 about T_a; (2) the canopy share of the available energy,
    #     A_c = A (1 - exp(-k LAI)) = A k L_eff with the Beer-Lambert owner of Farquhar and
    #     Medlyn (biosphere.photosynthesis.canopy_light_extinction); (3) the LEB-local wind
    #     floor biosphere.leaf_energy_balance.minimum_wind_speed; (4) a Monin-Obukhov
    #     stability correction of g_a for heat and momentum with the MOST owner
    #     relic.scientific.atmosphere.most (Businger-Dyer psi functions, integrated
    #     Paulson profile); the self-consistent zeta is bracketed on the owner's
    #     Businger-Dyer domain and found by config["stability_iterations"] bisection steps
    #     (dT -> H -> L -> zeta -> u*, g_a -> dT). The radiation inputs
    #     shortwave_net, longwave_down and surface_emissivity of the radiation owner replace
    #     net_radiation (RadiationFunctor: L_net = eps L_down - eps sigma T_s^4, so
    #     R_ni = R_n + eps sigma (T_s^4 - T_a^4)).
    ENERGY_BALANCE_FORMS = ("linearised_big_leaf", "radiative_canopy_share")
    # Fixed number of bisection steps of the stability (config "stability_iterations",
    # radiative_canopy_share only); the loop count never depends on the data. 40 steps close
    # the bracket [-2, 2] to 4 2^-40 (below float64 resolution of zeta near 1).
    DEFAULT_STABILITY_ITERATIONS = 40
    MAXIMUM_STABILITY_ITERATIONS = 60
    # Free-convection velocity of the stability-corrected wind (config "free_convection",
    # radiative_canopy_share only; absent or "none" keeps the wind u):
    #   "beljaars": U = sqrt(u^2 + U_c^2), U_c = beta w* for zeta < 0 and 0 otherwise,
    #     w* = (g H z_i / (rho c_p T_a))^(1/3) (CTSM Technical Note, Eqs. 2.5.24, 2.5.28,
    #     2.5.29; beta and z_i in biosphere.leaf_energy_balance).
    FREE_CONVECTION_FORMS = ("none", "beljaars")
    FREE_CONVECTION_OUTPUTS = ("diagnostics.leaf_energy_balance.convective_velocity",
                               "diagnostics.leaf_energy_balance.gusty_wind_speed")
    FREE_CONVECTION_PARAMETERS = {
        "biosphere.leaf_energy_balance.convective_velocity_coefficient": (
            "Coefficient beta of the convective velocity U_c = beta w* [dimensionless]"
        ),
        "biosphere.leaf_energy_balance.convective_boundary_layer_height": (
            "Convective boundary-layer height z_i of the velocity scale w* [m]"
        ),
    }
    RADIATIVE_INPUTS = ("shortwave_net", "longwave_down", "surface_emissivity")
    RADIATIVE_OUTPUTS = ("diagnostics.leaf_energy_balance.isothermal_net_radiation",
                         "diagnostics.leaf_energy_balance.canopy_energy_fraction",
                         "diagnostics.leaf_energy_balance.radiative_conductance",
                         "diagnostics.leaf_energy_balance.longwave_emission_change",
                         "diagnostics.leaf_energy_balance.stability_parameter",
                         "diagnostics.leaf_energy_balance.stability_iteration_residual")
    RADIATIVE_PARAMETERS = {
        "biosphere.photosynthesis.canopy_light_extinction": (
            "Beer-Lambert extinction coefficient k of the canopy share 1 - exp(-k LAI) "
            "[dimensionless]"
        ),
        "biosphere.leaf_energy_balance.minimum_wind_speed": (
            "Lower bound of the wind speed in the aerodynamic conductance of the balance [m s-1]"
        ),
        "atmosphere.most.stable_correction_limit": (
            "Upper zeta bound of the Businger-Dyer domain of the MOST owner, the cap of the "
            "stability correction [dimensionless]"
        ),
        "atmosphere.most.unstable_correction_limit": (
            "Absolute lower zeta bound of the Businger-Dyer domain of the MOST owner, the cap "
            "of the stability correction [dimensionless]"
        ),
    }

    def __init__(self, name: str = FUNCTOR_NAME, config: Optional[dict] = None,
                 *, parameter_manager: Any = None, device: Any = None) -> None:
        configuration = dict(config or {})
        temporal_support = configuration.get("temporal_support")
        if temporal_support not in self.TEMPORAL_SUPPORTS:
            raise ValueError(
                "LeafEnergyBalance config must declare temporal_support as one of "
                f"{list(self.TEMPORAL_SUPPORTS)}, got {temporal_support!r}"
            )
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                "Big-leaf canopy energy balance (linearised Penman-Monteith form): the "
                "leaf (canopy) temperature that closes R_n - G = H + lambda E with the "
                "FAO-56 Eq. 4 aerodynamic conductance shared with Penman-Monteith and the "
                "supplied canopy conductance; all of R_n - G is available to the big leaf; "
                "no radiative resistance (R_n is the supplied net radiation, not the "
                "isothermal one); leaf_area_index <= 0 declares no canopy and T_leaf = T_air."
            ),
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "dT = [gamma (g_a + g_c) A / (rho c_p) - g_a g_c D] / "
                "(g_a [s g_c + gamma (g_a + g_c)]), A = R_n - G; T_leaf = T_air + dT; "
                "H = rho c_p g_a dT; lambda E = A - H; daily: G = 0, "
                "T_leaf,max = T_max + dT"
            ),
            references=[
                "Monteith, J.L. (1965). Evaporation and environment. Symp. Soc. Exp. Biol. 19, 205-224",
                "Allen, R.G. et al. (1998). Crop evapotranspiration. FAO Irrigation and Drainage Paper 56, Eqs. 3, 4 and 42",
                "Jones, H.G. (2014). Plants and Microclimate, 3rd ed. Cambridge University Press, Chapter 9",
                "Campbell, G.S. and Norman, J.M. (1998). An Introduction to Environmental Biophysics, 2nd ed. Springer, Chapter 14",
                "Monteith, J.L. and Unsworth, M.H. (2013). Principles of Environmental Physics, 4th ed. Academic Press",
            ],
            parameters={
                "hydrosphere.evapotranspiration.roughness_length": (
                    "Momentum roughness length z0m of the logarithmic wind profile [m]"
                ),
                "hydrosphere.evapotranspiration.displacement_height": (
                    "Zero-plane displacement height d of the logarithmic wind profile [m]"
                ),
                "hydrosphere.evapotranspiration.reference_height": (
                    "Measurement height z of wind speed, temperature and humidity [m]"
                ),
                "hydrosphere.physical.heat_to_momentum_roughness_ratio": (
                    "Ratio z0h/z0m of the heat and momentum roughness lengths [dimensionless]"
                ),
                "hydrosphere.numerical.minimum_wind_speed": (
                    "Lower bound of the wind speed in the aerodynamic conductance [m s-1]"
                ),
                "hydrosphere.numerical.aerodynamic_conductance_bounds": (
                    "Bounds [min, max] of the aerodynamic conductance g_a [m s-1]"
                ),
                "hydrosphere.numerical.temperature_bounds": (
                    "Plausibility bounds [min, max] of the air temperature inputs [K]"
                ),
                "hydrosphere.numerical.net_radiation_bounds": (
                    "Plausibility bounds [min, max] of the net radiation input [W m-2]"
                ),
                "hydrosphere.physical.atmospheric_pressure_min": (
                    "Lower plausibility bound of the atmospheric pressure input [Pa]"
                ),
                "hydrosphere.physical.atmospheric_pressure_max": (
                    "Upper plausibility bound of the atmospheric pressure input [Pa]"
                ),
            },
            input_variables=["air_temperature", "vapor_pressure_deficit", "wind_speed",
                             "atmospheric_pressure", "net_radiation", "canopy_conductance",
                             "leaf_area_index"],
            output_variables=["leaf_temperature",
                              "diagnostics.leaf_energy_balance.leaf_air_temperature_difference",
                              "diagnostics.leaf_energy_balance.sensible_heat_flux",
                              "diagnostics.leaf_energy_balance.latent_heat_flux",
                              "diagnostics.leaf_energy_balance.available_energy",
                              "diagnostics.leaf_energy_balance.aerodynamic_conductance",
                              "diagnostics.leaf_energy_balance.no_canopy",
                              "diagnostics.leaf_energy_balance.vapor_pressure_floor_active"],
            conservation_laws=[],
            spatial_dimensions=0,
            spatial_scale=SpatialScale.PATCH,
            temporal_scale=TemporalScale.HOUR,
            variable_kind=VariableKind.DIAGNOSTIC,
            conflict_resolution_policy=ConflictResolutionPolicy.EXCLUSIVE,
        )
        energy_balance_form = configuration.get("energy_balance_form", "linearised_big_leaf")
        if energy_balance_form not in self.ENERGY_BALANCE_FORMS:
            raise ValueError(
                "LeafEnergyBalance energy_balance_form must be one of "
                f"{list(self.ENERGY_BALANCE_FORMS)}, got {energy_balance_form!r}"
            )
        radiative = energy_balance_form == "radiative_canopy_share"
        stability_iterations = configuration.get("stability_iterations")
        if stability_iterations is not None and not radiative:
            raise ValueError("LeafEnergyBalance stability_iterations needs energy_balance_form "
                             "'radiative_canopy_share'")
        if stability_iterations is None:
            stability_iterations = self.DEFAULT_STABILITY_ITERATIONS
        if (isinstance(stability_iterations, bool) or not isinstance(stability_iterations, int)
                or not 1 <= stability_iterations <= self.MAXIMUM_STABILITY_ITERATIONS):
            raise ValueError("LeafEnergyBalance stability_iterations must be an integer in "
                             f"[1, {self.MAXIMUM_STABILITY_ITERATIONS}], got {stability_iterations!r}")
        free_convection = configuration.get("free_convection")
        if free_convection is not None and not radiative:
            raise ValueError("LeafEnergyBalance free_convection needs energy_balance_form "
                             "'radiative_canopy_share'")
        free_convection = "none" if free_convection is None else free_convection
        if free_convection not in self.FREE_CONVECTION_FORMS:
            raise ValueError("LeafEnergyBalance free_convection must be one of "
                             f"{list(self.FREE_CONVECTION_FORMS)}, got {free_convection!r}")
        if free_convection == "beljaars":
            metadata.output_variables.extend(self.FREE_CONVECTION_OUTPUTS)
            metadata.parameters.update(self.FREE_CONVECTION_PARAMETERS)
        if radiative:
            index = metadata.input_variables.index("net_radiation")
            metadata.input_variables[index:index + 1] = list(self.RADIATIVE_INPUTS)
            metadata.output_variables.extend(self.RADIATIVE_OUTPUTS)
            metadata.parameters.update(self.RADIATIVE_PARAMETERS)
            metadata.parameters.pop("hydrosphere.numerical.minimum_wind_speed")
            metadata.mathematical_expression = (
                "radiative_canopy_share: R_ni = S_n + eps L_down - eps sigma T_a^4; "
                "A_c = (R_ni - G) (1 - exp(-k LAI)); g_R = 4 eps sigma T_a^3 / (rho c_p); "
                "dT = [gamma (g_a + g_c) A_c / (rho c_p) - g_a g_c D] / "
                "[gamma (g_a + g_R) (g_a + g_c) + s g_a g_c]; T_leaf = T_air + dT; "
                "H = rho c_p g_a dT; L = rho c_p g_R dT; lambda E = A_c - H - L; "
                "g_a = k u* / f_h(zeta), u* = k U / f_m(zeta), zeta = (z - d) / L_MO(H), "
                "zeta the bracketed root of zeta - Z(zeta) on the Businger-Dyer domain (bisection)")
        daylight_rule = None
        if temporal_support == "daily":
            metadata.input_variables.extend(self.DAILY_INPUTS)
            metadata.output_variables.extend(self.DAILY_OUTPUTS)
        else:
            metadata.input_variables.extend(self.INSTANTANEOUS_INPUTS)
        # The inputs and form of the balance law: the daylight support evaluates the
        # instantaneous form at its samples.
        law_inputs = list(metadata.input_variables)
        law_form = "daily" if temporal_support == "daily" else "instantaneous"
        if temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            daylight_rule = daylight_quadrature.daylight_config(configuration, "LeafEnergyBalance")
            metadata.input_variables[:] = [
                self.DAYLIGHT_REPLACED_INPUTS.get(key, key)
                for key in metadata.input_variables if key not in self.INSTANTANEOUS_INPUTS]
            metadata.input_variables.extend(
                key for key in self.DAYLIGHT_INPUTS if key not in metadata.input_variables)
            metadata.output_variables.extend(self.DAYLIGHT_OUTPUTS)
        else:
            daylight_quadrature.reject_daylight_keys(configuration, "LeafEnergyBalance")
        super().__init__(name=name, config=configuration, metadata=metadata,
                         parameter_manager=parameter_manager, device=device)
        self.temporal_support = temporal_support
        self._law_inputs = law_inputs
        self._law_form = law_form
        self._daylight_rule = daylight_rule
        self.energy_balance_form = energy_balance_form
        self._radiative = radiative
        self.stability_iterations = int(stability_iterations)
        self.free_convection = free_convection
        # References: CODATA 2018 via the physical-constant SSOT (as RadiationFunctor).
        # Mathematical expression: sigma of M = eps sigma T^4.
        # Parameter meanings: sigma the Stefan-Boltzmann constant.
        # Units: sigma [W m-2 K-4].
        # Domain/constraints: read only with radiative_canopy_share.
        # Sign convention: not applicable (constant).
        self._stefan_boltzmann = float(get_physical_constant("stefan_boltzmann_constant")) if radiative else None
        # References: physical-constant SSOT; Allen et al. (1998), FAO-56 Eqs. 8, 11, 13.
        # Mathematical expression: epsilon = M_w / M_d; lambda, c_p and the Tetens
        #   constants e0, a, b, c of e_s and s are read once, as PenmanMonteithET does.
        # Parameter meanings: lambda latent heat of vaporization; c_p specific heat of air
        #   at constant pressure; M_w, M_d molar masses of water and dry air; K0 the
        #   Celsius offset; dt_day the length of the daily support.
        # Units: lambda [J kg-1]; c_p [J kg-1 K-1]; M [g mol-1]; e0 [Pa]; a [1]; b [degC];
        #   c [degC]; K0 [K]; dt_day [s].
        # Domain/constraints: SSOT constants, positive.
        # Sign convention: not applicable (material constants).
        self.latent_heat_vaporization = float(get_physical_constant("latent_heat_vaporization"))
        self.specific_heat_air = float(get_physical_constant("air_specific_heat_cp"))
        self.molecular_weight_ratio = (float(get_physical_constant("water_molecular_weight"))
                                       / float(get_physical_constant("air_molecular_weight")))
        self._tetens = {
            "tetens_base_pressure": float(get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_base_pressure")),
            "tetens_coeff_a": float(get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_coefficient_a")),
            "tetens_coeff_b": float(get_physical_constant(
                "relic.physical.constants.psychrometrics.tetens_coefficient_b")),
            "kelvin_offset": float(get_physical_constant("celsius_to_kelvin")),
        }
        self._tetens_slope_coeff = float(get_physical_constant(
            "relic.physical.constants.psychrometrics.saturation_vapor_pressure_slope_coeff"))
        self.von_karman_constant = float(get_physical_constant("von_karman_constant"))
        self._seconds_per_day = float(get_physical_constant("seconds_per_day"))

    def get_dependencies(self) -> list:
        """Declare no Functor dependency; every input is a state port.

        canopy_conductance (medlyn_stomatal_conductance or a forcing) and net_radiation
        (atmospheric_radiation or a forcing) are bound by graph links or supplied as
        state, so the static declaration is empty; instance declarations are kept
        (FunctorBase._merge_declared_dependencies).
        """
        return self._merge_declared_dependencies([])

    def _input(self, state: StateDict, key: str, like: Optional[Tensor] = None) -> Tensor:
        if state.get(key) is None:
            raise ValueError(f"LeafEnergyBalance requires the declared input {key!r}")
        value = state[key]
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value, dtype=like.dtype if like is not None else None)
        elif like is not None:
            value = value.to(dtype=like.dtype)
        if not value.is_floating_point():
            raise TypeError(f"{key} must use a floating tensor dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{key} must be finite")
        return value

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References: Goudriaan (1986), Agric. For. Meteorol. 38, 249-254;
        #   relic.scientific.atmosphere.daylight_quadrature (daylight_gauss support).
        # Mathematical expression: daily and instantaneous supports: _balance_law on the
        #   inputs; daylight_gauss: X = sum_k W_k X_k with X_k the instantaneous balance
        #   of sample k (_forward_daylight_gauss).
        # Parameter meanings: W_k the daylight sample weights.
        # Units: as _balance_law.
        # Domain/constraints: the daily and daylight supports require dt = seconds_per_day.
        # Sign convention: as _balance_law.
        if self.temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            return self._forward_daylight_gauss(state, time, dt, **kwargs)
        return self._balance_law(state, time, dt, **kwargs)

    def _forward_daylight_gauss(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """Daily means of the instantaneous balance over the daylight Gauss samples."""
        # References: Jones (2014), Chapter 9; Goudriaan (1986); Goudriaan and van Laar
        #   (1994); daylight_quadrature (FAO-56 geometry, SWAT+ Eq. 1:1.3.1 course).
        # Mathematical expression: R_n,k = R_n - S_n + S_n,k with S_n,k the daily net
        #   shortwave S_n redistributed by the solar elevation (sum_k W_k S_n,k = S_n, so
        #   sum_k W_k R_n,k = R_n); G_k = 0; dT_k, T_l,k, H_k, lambda E_k of the
        #   instantaneous balance with T_k, D_k = max(e0(T_k) - e_a, 0) and g_c,k;
        #   X = sum_k W_k X_k; T_l,max = T_max + dT_m with m the daylight instant of the
        #   highest sample air temperature (T_l,max = T_max where LAI <= 0).
        # Parameter meanings: R_n the 24-hour mean net radiation; S_n its net shortwave
        #   part; g_c,k the Medlyn canopy conductance of sample k (night: the night value);
        #   W_k the sample weights; wind, pressure and LAI held constant over the day.
        # Units: R, S, H, lambda E [W m-2]; T [K]; D, e [Pa]; g [m s-1].
        # Domain/constraints: dt = seconds_per_day; the input checks of the balance apply
        #   to every sample. Declared approximations: the net longwave is constant over
        #   the day and G_k = 0 at every instant (FAO-56 Eq. 42 holds for the daily sum
        #   only, so daytime available energy is overestimated by the daytime soil heat
        #   flux); T_l,max is a declared estimate, exact only if the air maximum fell on
        #   the selected instant.
        # Sign convention: as _balance_law.
        daylight_quadrature.require_daily_step(dt, "LeafEnergyBalance")
        node_count, shortwave_shape = self._daylight_rule
        samples_key = "canopy_conductance_daylight_samples"
        air_temperature = self._input(state, "air_temperature")
        conductance = self._input(state, samples_key, like=air_temperature)
        values = {key: self._input(state, key, like=air_temperature)
                  for key in self.metadata.input_variables if key != samples_key}
        samples = daylight_quadrature.daylight_sample_state(
            values, {"canopy_conductance": conductance}, node_count=node_count,
            shortwave_shape=shortwave_shape, flux_keys=("shortwave_net",), temperature=True,
            vapor_pressure_deficit=True, dtype=air_temperature.dtype)
        sample_state = dict(samples.state)
        support = samples.support
        if self._radiative:
            # radiative_canopy_share: S_n,k redistributed, L_down and eps constant over the
            #   day; the law forms R_ni,k = S_n,k + eps L_down - eps sigma T_k^4 itself.
            sample_state["soil_heat_flux"] = torch.zeros_like(sample_state["shortwave_net"])
        else:
            net_shortwave = sample_state.pop("shortwave_net")
            sample_state["net_radiation"] = (
                sample_state["net_radiation"] - support.constant(values["shortwave_net"]) + net_shortwave)
            sample_state["soil_heat_flux"] = torch.zeros_like(sample_state["net_radiation"])
        sample_outputs = self._balance_law(sample_state, time, dt, **kwargs)
        outputs = daylight_quadrature.daily_outputs(sample_outputs, support)
        leaf_samples = sample_outputs["leaf_temperature"]
        outputs["leaf_temperature_daylight_samples"] = UpdateItem(
            value=leaf_samples.value, mode=UpdateMode.REPLACE, unit="K", per_timestep=False,
            source_functor=self.name)
        difference = sample_outputs[self.DIAGNOSTIC_PREFIX + "leaf_air_temperature_difference"]
        daylight_air = sample_state["air_temperature"][..., :node_count]
        warmest = torch.argmax(daylight_air, dim=-1, keepdim=True)
        peak_difference = torch.gather(difference[..., :node_count], -1, warmest).squeeze(-1)
        air_temperature_max = values["air_temperature_max"]
        no_canopy = outputs[self.DIAGNOSTIC_PREFIX + "no_canopy"]
        peak_difference, air_temperature_max, no_canopy = torch.broadcast_tensors(
            peak_difference, air_temperature_max, no_canopy)
        outputs["leaf_temperature_max"] = UpdateItem(
            value=torch.where(no_canopy, air_temperature_max, air_temperature_max + peak_difference),
            mode=UpdateMode.REPLACE, unit="K", per_timestep=False, source_functor=self.name)
        return outputs

    def _balance_law(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References: Monteith (1965); Allen et al. (1998), FAO-56 Eqs. 3, 4, 8, 11, 13
        #   and 42; Jones (2014), Chapter 9; Campbell and Norman (1998), Chapter 14.
        # Mathematical expression: A = R_n - G; e_a = e_s - D (e_s of FAO-56 Eq. 12 daily);
        #   rho = rho(T_a, p, e_a); dT = [gamma (g_a + g_c) A / (rho c_p) - g_a g_c D] /
        #   (g_a [s g_c + gamma (g_a + g_c)]); T_l = T_a + dT; H = rho c_p g_a dT;
        #   lambda E = A - H; T_l,max = T_max + dT (daily).
        # Parameter meanings: T_a air temperature (the linearisation point of e_s); D the
        #   vapour pressure deficit of the air; s = de_s/dT at T_a; gamma = c_p p /
        #   (epsilon lambda); g_a the FAO-56 Eq. 4 conductance shared with
        #   PenmanMonteithET; g_c the bulk canopy conductance (Medlyn g_s L_eff R T / P);
        #   R_n the net radiation of the surface, all of it available to the big leaf
        #   (canopy fraction 1, the single-source assumption of Penman-Monteith).
        # Units: T [K]; D, e [Pa]; s, gamma [Pa K-1]; rho [kg m-3]; c_p [J kg-1 K-1];
        #   g [m s-1]; R_n, G, A, H, lambda E [W m-2].
        # Domain/constraints: finite inputs; T_a (and T_max, T_min) inside
        #   hydrosphere.numerical.temperature_bounds and T_min <= T_a <= T_max; R_n inside
        #   net_radiation_bounds; p inside [atmospheric_pressure_min, _max]; D >= 0;
        #   wind >= 0; g_c >= 0 (g_c = 0 is the dry-leaf limit, not an error);
        #   LAI <= 0 is no canopy: T_l = T_a, dT = H = lambda E = 0 and T_l,max = T_max
        #   there. e_a < 0 (D > e_s(T_a), possible for daily means of very dry days) is
        #   replaced by 0 in rho only and flagged in vapor_pressure_floor_active. The
        #   denominator is >= gamma g_a^2 > 0 and is checked, never clamped; no output
        #   is clipped. Linearisation error grows with |dT| (second-order term
        #   0.5 s' dT^2 neglected).
        # Sign convention: R_n and A positive into the surface; G positive into the
        #   soil; H and lambda E positive from the leaf to the air; dT > 0 a leaf
        #   warmer than the air.
        if self._law_form == "daily" and float(dt) != self._seconds_per_day:
            raise ValueError(
                f"LeafEnergyBalance daily support requires dt={self._seconds_per_day} s, "
                f"got dt={float(dt)} s (FAO-56 Eq. 42 is a 24-hour form)"
            )
        air_temperature = self._input(state, "air_temperature")
        values: Dict[str, Tensor] = {"air_temperature": air_temperature}
        for key in self._law_inputs:
            if key != "air_temperature":
                values[key] = self._input(state, key, like=air_temperature)
        support_shapes = {tuple(value.shape) for value in values.values() if value.ndim}
        if len(support_shapes) > 1:
            raise ValueError("LeafEnergyBalance non-scalar input shapes must share one spatial support.")
        values = dict(zip(values, torch.broadcast_tensors(*values.values()), strict=True))
        air_temperature = values["air_temperature"]
        vpd = values["vapor_pressure_deficit"]
        wind_speed = values["wind_speed"]
        pressure = values["atmospheric_pressure"]
        if self._radiative:
            net_radiation = self._isothermal_net_radiation(values)
        else:
            net_radiation = values["net_radiation"]
        canopy_conductance = values["canopy_conductance"]
        leaf_area_index = values["leaf_area_index"]

        temp_bounds = self.get_parameter_value(
            "hydrosphere.numerical.temperature_bounds", required=True)
        validate_temperature_range(air_temperature, min_temp=float(temp_bounds[0]),
                                   max_temp=float(temp_bounds[1]))
        rn_bounds = self.get_parameter_value(
            "hydrosphere.numerical.net_radiation_bounds", required=True)
        if bool(torch.any(net_radiation < float(rn_bounds[0]))) or bool(
                torch.any(net_radiation > float(rn_bounds[1]))):
            raise ValueError(
                f"net_radiation outside physical bounds [{rn_bounds[0]}, {rn_bounds[1]}] W m-2")
        pressure_min = float(self.get_parameter_value(
            "hydrosphere.physical.atmospheric_pressure_min", required=True))
        pressure_max = float(self.get_parameter_value(
            "hydrosphere.physical.atmospheric_pressure_max", required=True))
        if bool(torch.any(pressure < pressure_min)) or bool(torch.any(pressure > pressure_max)):
            raise ValueError(
                f"atmospheric_pressure outside [{pressure_min}, {pressure_max}] Pa")
        if bool(torch.any(vpd < 0.0)):
            raise ValueError("vapor_pressure_deficit must be non-negative")
        if bool(torch.any(wind_speed < 0.0)):
            raise ValueError("wind_speed must be non-negative")
        if bool(torch.any(canopy_conductance < 0.0)):
            raise ValueError("canopy_conductance must be non-negative")
        if bool(torch.any(leaf_area_index < 0.0)):
            raise ValueError("leaf_area_index must be non-negative")
        if self._law_form == "daily":
            for key in self.DAILY_INPUTS:
                validate_temperature_range(values[key], min_temp=float(temp_bounds[0]),
                                           max_temp=float(temp_bounds[1]))
            if bool(torch.any(values["air_temperature_min"] > air_temperature)) or bool(
                    torch.any(values["air_temperature_max"] < air_temperature)):
                raise ValueError(
                    "LeafEnergyBalance daily support requires air_temperature_min <= "
                    "air_temperature <= air_temperature_max")

        # References: Allen et al. (1998), FAO-56, Eq. 42 (daily G = 0); otherwise the
        #   declared soil_heat_flux input.
        # Mathematical expression: A = R_n - G.
        # Parameter meanings: R_n net radiation of the surface; G soil heat flux.
        # Units: A, R_n, G [W m-2].
        # Domain/constraints: daily support only at dt = seconds_per_day (checked above).
        # Sign convention: A positive into the big leaf; G positive into the soil.
        if self._law_form == "daily":
            soil_heat_flux = torch.zeros_like(net_radiation)
        else:
            soil_heat_flux = values["soil_heat_flux"]
        available_energy = net_radiation - soil_heat_flux
        if self._radiative:
            # References: Monsi and Saeki (1953); de Pury and Farquhar (1997) (the
            #   Beer-Lambert owner beer_lambert_effective_leaf_area shared with Farquhar and
            #   Medlyn); biosphere.photosynthesis.canopy_light_extinction.
            # Mathematical expression: f_c = k L_eff = 1 - exp(-k LAI), L_eff = (1 -
            #   exp(-k LAI)) / k; A_c = f_c A.
            # Parameter meanings: f_c the fraction of the available energy intercepted by the
            #   canopy (the attenuation of the radiation through LAI with the extinction k of
            #   the photosynthesis canopy); A_c the available energy of the big leaf; the rest
            #   (1 - f_c) A belongs to the soil surface, which this Functor does not model.
            # Units: k [1]; LAI, L_eff [m2 m-2]; f_c [1]; A, A_c [W m-2].
            # Domain/constraints: LAI >= 0 (checked); f_c in [0, 1); f_c -> 0 as LAI -> 0, so
            #   dT -> 0 continuously; the same k is applied to the net longwave part of A
            #   (one extinction for the whole available energy, a declared simplification).
            # Sign convention: as A.
            extinction = self._ensure_tensor(self.get_parameter_value(
                "biosphere.photosynthesis.canopy_light_extinction", required=True), dtype=leaf_area_index.dtype)
            canopy_fraction = extinction * beer_lambert_effective_leaf_area(leaf_area_index, extinction)
            available_energy = available_energy * canopy_fraction

        # References: Allen et al. (1998), FAO-56 Eqs. 8, 11 and 13 (shared helpers of
        #   hydrosphere.utils.psychrometrics); Holton and Hakim (2012) Eqs. 2.63-2.64
        #   (moist-air density, compute_moist_air_density).
        # Mathematical expression: e0(T) = e0_ref exp(a Tc / (Tc + b)); s = c e0(T_a) /
        #   (Tc + b)^2; gamma = c_p p / (epsilon lambda); e_s = e0(T_a), or on the daily
        #   support e_s = (e0(T_max) + e0(T_min)) / 2 (FAO-56 Eq. 12, as PenmanMonteithET);
        #   e_a = max(e_s - D, 0) (density only); rho = (p - (1 - epsilon) e_a) / (R_d T_a).
        # Parameter meanings: as in PenmanMonteithET; the same helpers and constants.
        # Units: e_s, e_a, p [Pa]; s, gamma [Pa K-1]; rho [kg m-3]; Tc [degC].
        # Domain/constraints: T_a inside the validated bounds, so Tc + b > 0 and s > 0;
        #   p > 0, so gamma > 0; the e_a floor is flagged, not silent. s is the slope at
        #   T_a, the linearisation point of e_s(T_l).
        # Sign convention: positive meteorological properties.
        saturation_air = compute_saturation_vapor_pressure(air_temperature, **self._tetens)
        slope = compute_slope_saturation_vapor_pressure(
            air_temperature, es_pa=saturation_air, tetens_coeff_b=self._tetens["tetens_coeff_b"],
            tetens_slope_coeff=self._tetens_slope_coeff,
            kelvin_offset=self._tetens["kelvin_offset"])
        if self._law_form == "daily":
            saturation = 0.5 * (
                compute_saturation_vapor_pressure(values["air_temperature_max"], **self._tetens)
                + compute_saturation_vapor_pressure(values["air_temperature_min"], **self._tetens))
        else:
            saturation = saturation_air
        gamma = compute_psychrometric_constant(
            pressure, specific_heat_air=self.specific_heat_air,
            molecular_weight_ratio=self.molecular_weight_ratio,
            latent_heat=self.latent_heat_vaporization)
        actual_vapor_pressure = saturation - vpd
        vapor_pressure_floor = actual_vapor_pressure < 0.0
        actual_vapor_pressure = torch.where(
            vapor_pressure_floor, torch.zeros_like(actual_vapor_pressure), actual_vapor_pressure)
        density = compute_moist_air_density(air_temperature, pressure, actual_vapor_pressure)
        volumetric_heat = density * self.specific_heat_air  # J m-3 K-1

        # References: Allen et al. (1998), FAO-56 Eq. 4 (fao56_aerodynamic_conductance,
        #   the owner shared with PenmanMonteithET).
        # Mathematical expression: g_a = k^2 max(u, u_min) / (ln((z - d)/z0m)
        #   ln((z - d)/z0h)), limited to aerodynamic_conductance_bounds.
        # Parameter meanings: z, d, z0m, z0h = c_h z0m the profile heights; k von Karman.
        # Units: g_a, u [m s-1]; heights [m].
        # Domain/constraints: neutral stability; heat and vapour share r_a = 1 / g_a.
        # Sign convention: positive transfer magnitude.
        aerodynamic_conductance = fao56_aerodynamic_conductance(
            wind_speed,
            reference_height=self.get_parameter_value(
                "hydrosphere.evapotranspiration.reference_height", required=True),
            displacement_height=self.get_parameter_value(
                "hydrosphere.evapotranspiration.displacement_height", required=True),
            roughness_length=self.get_parameter_value(
                "hydrosphere.evapotranspiration.roughness_length", required=True),
            heat_to_momentum_ratio=self.get_parameter_value(
                "hydrosphere.physical.heat_to_momentum_roughness_ratio", required=True),
            von_karman_constant=self.von_karman_constant,
            minimum_wind_speed=self.get_parameter_value(
                "biosphere.leaf_energy_balance.minimum_wind_speed" if self._radiative
                else "hydrosphere.numerical.minimum_wind_speed", required=True),
            conductance_bounds=self.get_parameter_value(
                "hydrosphere.numerical.aerodynamic_conductance_bounds", required=True),
        )
        if self._radiative:
            return self._radiative_balance(
                values, air_temperature=air_temperature, vpd=vpd, net_radiation=net_radiation,
                available_energy=available_energy, canopy_fraction=canopy_fraction,
                canopy_conductance=canopy_conductance, leaf_area_index=leaf_area_index,
                slope=slope, gamma=gamma, volumetric_heat=volumetric_heat,
                aerodynamic_conductance=aerodynamic_conductance,
                vapor_pressure_floor=vapor_pressure_floor, density=density,
                wind_speed=wind_speed)

        # References: Jones (2014), Chapter 9, the linearised leaf energy balance with
        #   r_HR = r_aW = r_a; Monteith (1965); Allen et al. (1998), FAO-56 Eq. 3.
        # Mathematical expression: N = gamma (g_a + g_c) A / (rho c_p) - g_a g_c D;
        #   Q = g_a [s g_c + gamma (g_a + g_c)]; dT = N / Q.
        # Parameter meanings: as above; Q is the resistance-form denominator
        #   gamma (r_a + r_c) + s r_a times g_a^2 g_c, with r_a = 1/g_a, r_c = 1/g_c.
        # Units: N [K m s-1 Pa K-1]; Q [m2 s-2 Pa K-1]; dT [K].
        # Domain/constraints: Q >= gamma g_a^2 > 0 for g_a > 0, gamma > 0, s > 0 and
        #   g_c >= 0; a non-finite or non-positive Q is a violated input contract and
        #   raises. g_c = 0 gives dT = A / (rho c_p g_a) without a division by g_c.
        # Sign convention: dT > 0 a leaf warmer than the air.
        numerator = (gamma * (aerodynamic_conductance + canopy_conductance)
                     * available_energy / volumetric_heat
                     - aerodynamic_conductance * canopy_conductance * vpd)
        denominator = aerodynamic_conductance * (
            slope * canopy_conductance + gamma * (aerodynamic_conductance + canopy_conductance))
        if not bool(torch.all(torch.isfinite(denominator) & (denominator > 0.0))):
            raise ValueError(
                "LeafEnergyBalance denominator g_a [s g_c + gamma (g_a + g_c)] must be finite "
                "and positive; check air_temperature, atmospheric_pressure and the conductances")
        difference = numerator / denominator

        # References: explicit configuration of the big-leaf support (no leaves, no leaf
        #   temperature), as nonpositive_leaf_area "no_canopy" of MedlynStomatalConductance.
        # Mathematical expression: N_c = (LAI <= 0); dT = 0, H = lambda E = 0 where N_c.
        # Parameter meanings: N_c the no-canopy mask.
        # Units: LAI [m2 m-2]; N_c [1].
        # Domain/constraints: the energy of a no-canopy cell belongs to the soil surface,
        #   which this Functor does not model; T_leaf = T_air is a declared placeholder of
        #   an absent leaf, not a soil temperature.
        # Sign convention: not applicable (mask).
        no_canopy = leaf_area_index <= 0.0
        zero = torch.zeros_like(difference)
        difference = torch.where(no_canopy, zero, difference)
        sensible_heat = torch.where(no_canopy, zero, volumetric_heat * aerodynamic_conductance * difference)
        latent_heat = torch.where(no_canopy, zero, available_energy - sensible_heat)
        leaf_temperature = torch.where(no_canopy, air_temperature, air_temperature + difference)
        if not bool(torch.isfinite(leaf_temperature).all()):
            raise ValueError("LeafEnergyBalance leaf_temperature overflowed the input precision")

        prefix = self.DIAGNOSTIC_PREFIX
        outputs: Dict[str, Any] = {
            "leaf_temperature": UpdateItem(value=leaf_temperature, mode=UpdateMode.REPLACE,
                                           unit="K", per_timestep=False, source_functor=self.name),
            prefix + "leaf_air_temperature_difference": difference,
            prefix + "sensible_heat_flux": sensible_heat,
            prefix + "latent_heat_flux": latent_heat,
            prefix + "available_energy": available_energy,
            prefix + "aerodynamic_conductance": aerodynamic_conductance,
            prefix + "no_canopy": no_canopy,
            prefix + "vapor_pressure_floor_active": vapor_pressure_floor,
        }
        if self._law_form == "daily":
            # References: declared estimate of this Functor (no literature law); the shift
            #   of the daily extreme by the daily-mean balance.
            # Mathematical expression: T_l,max = T_max + dT, dT of the 24-hour mean balance.
            # Parameter meanings: T_max the daily maximum air temperature.
            # Units: T [K].
            # Domain/constraints: T_max inside temperature_bounds and T_max >= T_a (checked
            #   above). Exact only if dT were constant over the day; the daytime dT at the
            #   time of T_max exceeds the 24-hour mean dT where daytime available energy
            #   exceeds the daily mean, so this estimate is biased low there. No canopy:
            #   T_l,max = T_max.
            # Sign convention: not applicable (temperature).
            air_temperature_max = values["air_temperature_max"]
            leaf_temperature_max = torch.where(
                no_canopy, air_temperature_max, air_temperature_max + difference)
            outputs["leaf_temperature_max"] = UpdateItem(
                value=leaf_temperature_max, mode=UpdateMode.REPLACE, unit="K",
                per_timestep=False, source_functor=self.name)
        return outputs

    def _isothermal_net_radiation(self, values: Dict[str, Tensor]) -> Tensor:
        """Isothermal net radiation R_ni of the radiative_canopy_share form [W m-2]."""
        # References: Stefan-Boltzmann law M = eps sigma T^4 (sigma of the CODATA 2018 SSOT
        #   constant); Kirchhoff's law (absorptivity = emissivity in the thermal band); the
        #   surface longwave budget of the radiation owner (RadiationFunctor, radiation.py:
        #   L_up = eps sigma T_s^4 + (1 - eps) L_down, L_net = L_down - L_up). The isothermal
        #   net radiation and the radiative resistance r_R = rho c_p / (4 eps sigma T_a^3)
        #   are the form presented by Jones (2014) and Campbell and Norman (1998); those books
        #   were not read here, so the form is derived in this comment and no chapter or
        #   equation number of them is quoted for it.
        # Mathematical expression: R_ni = S_n + eps L_down - eps sigma T_a^4, i.e. the net
        #   radiation of the radiation owner with the surface emission evaluated at the air
        #   temperature: R_ni = R_n + eps sigma (T_s^4 - T_a^4).
        # Parameter meanings: S_n the net shortwave (shortwave_net); L_down the downwelling
        #   longwave (longwave_down); eps the surface emissivity of the radiation owner
        #   (surface_emissivity, also the thermal absorptivity); T_a the air temperature.
        # Units: S_n, L_down, R_ni [W m-2]; eps [1]; sigma [W m-2 K-4]; T [K].
        # Domain/constraints: 0 < eps <= 1, L_down >= 0, S_n finite; R_ni is checked
        #   against net_radiation_bounds as R_n is in the other form.
        # Sign convention: R_ni positive into the surface.
        emissivity = values["surface_emissivity"]
        longwave_down = values["longwave_down"]
        if bool(torch.any(emissivity <= 0.0)) or bool(torch.any(emissivity > 1.0)):
            raise ValueError("surface_emissivity must lie in (0, 1]")
        if bool(torch.any(longwave_down < 0.0)):
            raise ValueError("longwave_down must be non-negative")
        emission = emissivity * self._stefan_boltzmann * torch.pow(values["air_temperature"], 4)
        return values["shortwave_net"] + emissivity * longwave_down - emission

    def _radiative_balance(self, values: Dict[str, Tensor], *, air_temperature: Tensor, vpd: Tensor,
                           net_radiation: Tensor, available_energy: Tensor, canopy_fraction: Tensor,
                           canopy_conductance: Tensor, leaf_area_index: Tensor, slope: Tensor,
                           gamma: Tensor, volumetric_heat: Tensor, aerodynamic_conductance: Tensor,
                           vapor_pressure_floor: Tensor, density: Tensor, wind_speed: Tensor) -> FluxDict:
        """Linearised balance with radiative coupling on the canopy share (radiative_canopy_share)."""
        # References: Monteith (1965) and Allen et al. (1998), FAO-56 Eq. 3 (combination
        #   form, unchanged vapour path); Stefan-Boltzmann law with a first-order Taylor
        #   expansion; Jones (2014) and Campbell and Norman (1998) present the same form (not
        #   read here; derivation below).
        # Mathematical expression (derivation): the leaf at T_l = T_a + dT absorbs R_ni and
        #   emits eps sigma T_l^4 instead of eps sigma T_a^4; eps sigma T_l^4 = eps sigma
        #   T_a^4 + 4 eps sigma T_a^3 dT + O(dT^2), so the extra emission is rho c_p g_R dT
        #   with g_R = 4 eps sigma T_a^3 / (rho c_p). The balance of the canopy share
        #   A_c = H + lambda E + rho c_p g_R dT with H = rho c_p g_a dT and lambda E =
        #   rho c_p g_v (s dT + D) / gamma, g_v = g_a g_c / (g_a + g_c) (aerodynamic and
        #   canopy conductances in series, e_s(T_l) linearised about T_a) gives
        #   dT = [gamma (g_a + g_c) A_c / (rho c_p) - g_a g_c D] /
        #        [gamma g_HR (g_a + g_c) + s g_a g_c], g_HR = g_a + g_R.
        #   With g_R = 0 and f_c = 1 it is the linearised_big_leaf form of this Functor.
        # Parameter meanings: A_c = f_c (R_ni - G) the canopy share of the isothermal
        #   available energy; g_a the FAO-56 Eq. 4 conductance (LEB wind floor); g_c the
        #   canopy conductance; g_R the radiative conductance; s, gamma, D, rho c_p as in
        #   _balance_law; L = rho c_p g_R dT the change of the leaf longwave emission.
        # Units: g [m s-1]; A_c, H, lambda E, L [W m-2]; dT [K]; sigma [W m-2 K-4].
        # Domain/constraints: the denominator is >= gamma g_HR g_a > 0 and is checked; no
        #   output is clipped; LAI <= 0 is no canopy (dT = H = lambda E = L = 0, T_l = T_a).
        #   Closure of the linearised balance: H + lambda E + L = A_c exactly.
        # Sign convention: dT > 0 a leaf warmer than the air; H, lambda E, L positive from
        #   the leaf to the air.
        emissivity = values["surface_emissivity"]
        radiative_conductance = (4.0 * emissivity * self._stefan_boltzmann
                                 * torch.pow(air_temperature, 3) / volumetric_heat)

        def leaf_air_difference(conductance: Tensor) -> Tensor:
            heat_conductance = conductance + radiative_conductance
            numerator = (gamma * (conductance + canopy_conductance)
                         * available_energy / volumetric_heat
                         - conductance * canopy_conductance * vpd)
            denominator = (gamma * heat_conductance * (conductance + canopy_conductance)
                           + slope * conductance * canopy_conductance)
            if not bool(torch.all(torch.isfinite(denominator) & (denominator > 0.0))):
                raise ValueError(
                    "LeafEnergyBalance denominator gamma g_HR (g_a + g_c) + s g_a g_c must be finite "
                    "and positive; check air_temperature, atmospheric_pressure and the conductances")
            return numerator / denominator

        difference, aerodynamic_conductance, stability, residual, gust = self._stability_corrected(
            leaf_air_difference, neutral_conductance=aerodynamic_conductance, wind_speed=wind_speed,
            air_temperature=air_temperature, density=density)
        no_canopy = leaf_area_index <= 0.0
        zero = torch.zeros_like(difference)
        difference = torch.where(no_canopy, zero, difference)
        sensible_heat = torch.where(no_canopy, zero, volumetric_heat * aerodynamic_conductance * difference)
        emission_change = torch.where(no_canopy, zero, volumetric_heat * radiative_conductance * difference)
        latent_heat = torch.where(no_canopy, zero, available_energy - sensible_heat - emission_change)
        leaf_temperature = torch.where(no_canopy, air_temperature, air_temperature + difference)
        if not bool(torch.isfinite(leaf_temperature).all()):
            raise ValueError("LeafEnergyBalance leaf_temperature overflowed the input precision")
        prefix = self.DIAGNOSTIC_PREFIX
        outputs: Dict[str, Any] = {
            "leaf_temperature": UpdateItem(value=leaf_temperature, mode=UpdateMode.REPLACE,
                                           unit="K", per_timestep=False, source_functor=self.name),
            prefix + "leaf_air_temperature_difference": difference,
            prefix + "sensible_heat_flux": sensible_heat,
            prefix + "latent_heat_flux": latent_heat,
            prefix + "available_energy": torch.where(no_canopy, zero, available_energy),
            prefix + "aerodynamic_conductance": aerodynamic_conductance,
            prefix + "no_canopy": no_canopy,
            prefix + "vapor_pressure_floor_active": vapor_pressure_floor,
            prefix + "isothermal_net_radiation": net_radiation,
            prefix + "canopy_energy_fraction": canopy_fraction,
            prefix + "radiative_conductance": radiative_conductance,
            prefix + "longwave_emission_change": emission_change,
            prefix + "stability_parameter": stability,
            prefix + "stability_iteration_residual": torch.where(no_canopy, zero, residual),
            **{prefix + name: value for name, value in gust.items()},
        }
        if self._law_form == "daily":
            # Declared estimate as in _balance_law: T_l,max = T_max + dT of the 24-hour balance.
            air_temperature_max = values["air_temperature_max"]
            outputs["leaf_temperature_max"] = UpdateItem(
                value=torch.where(no_canopy, air_temperature_max, air_temperature_max + difference),
                mode=UpdateMode.REPLACE, unit="K", per_timestep=False, source_functor=self.name)
        return outputs

    def _stability_corrected(self, leaf_air_difference: Any, *, neutral_conductance: Tensor,
                             wind_speed: Tensor, air_temperature: Tensor,
                             density: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
        """Monin-Obukhov correction of g_a by a fixed-count bisection; returns dT, g_a, zeta, residual."""
        # References: Monin and Obukhov (1954); Businger et al. (1971); Dyer (1974);
        #   Paulson (1970), the references of the RELIC MOST owner
        #   relic.scientific.atmosphere.most (friction_velocity, monin_obukhov_length, zeta,
        #   aerodynamic_resistance and the Businger-Dyer psi_m, psi_h it calls).
        # Mathematical expression: for a trial zeta, u*(zeta) = k U / f_m(zeta) and g_a(zeta) =
        #   k u* / f_h(zeta) (owner: f_x = ln((z - d)/z0x) - psi_x(zeta) + psi_x(zeta z0x/(z - d)),
        #   the integrated Paulson profile), clipped to aerodynamic_conductance_bounds;
        #   dT(zeta) = balance(g_a(zeta)); H = rho c_p g_a dT; L = -rho c_p T_a u*^3 / (k g H)
        #   (owner, with its |H| safeguard and L clamp); Z(zeta) = clip((z - d)/L, -zeta_u, zeta_s).
        #   The self-consistent stability is a root of F(zeta) = zeta - Z(zeta) on
        #   [-zeta_u, zeta_s]: F(-zeta_u) <= 0 <= F(zeta_s) because Z is clipped to the same
        #   interval, so the root is bracketed in every cell. N bisection steps halve the bracket
        #   (width (zeta_u + zeta_s) 2^-N); zeta* = bracket midpoint. A final fixed-point step
        #   zeta = Z(zeta*) gives g_a and dT = balance(g_a); residual |dT - dT(zeta*)|.
        #   With zeta = 0 the owner gives g_a = k^2 U / (ln((z - d)/z0m) ln((z - d)/z0h)), the
        #   FAO-56 Eq. 4 conductance (neutral limit).
        # Parameter meanings: U = max(u, u_min) with the LEB wind floor; z, d, z0m and
        #   z0h = c_h z0m the profile of FAO-56 Eq. 4 (hydrosphere.evapotranspiration); H the
        #   canopy sensible heat of the balance (the big leaf is the only heat source of the
        #   profile; soil sensible heat is not modelled); zeta_s and zeta_u the owner's
        #   Businger-Dyer domain limits atmosphere.most.stable_correction_limit and
        #   unstable_correction_limit (2 and 2), the bracket and cap of zeta (the owner
        #   refuses values outside them); N = config stability_iterations, fixed (no
        #   data-dependent loop). The plain and the under-relaxed fixed-point iterations were
        #   rejected: they cycle across the stable/unstable switch on real G050 cells with
        #   calm wind and small g_c (design note), where the bracket cannot.
        # Units: U, u*, g_a [m s-1]; z, d, z0, L [m]; H [W m-2]; zeta [1]; dT [K].
        # Domain/constraints: on the daily support the 24-hour mean dT drives zeta (a declared
        #   approximation: the diurnal course of stability is not resolved); on the daylight
        #   support every sample has its own bracket with the daily wind. The bisection runs
        #   without gradient; a final implicit-function step carries the derivative of the root,
        #   dzeta*/dtheta = (dZ/dtheta) / (1 - dZ/dzeta), with dZ/dzeta by a central difference
        #   (step 1e-4) and 1 - dZ/dzeta floored at 1e-3.
        #   Where H changes sign the owner's |H| safeguard makes Z jump by 2 (z - d) / 1000 m;
        #   the bracket then closes on the jump and the residual stays bounded by that jump.
        # Sign convention: dT > 0 gives H > 0, L < 0 and zeta < 0 (unstable, larger g_a);
        #   dT < 0 gives zeta > 0 (stable, smaller g_a).
        dtype, device = air_temperature.dtype, air_temperature.device

        def number(key: str) -> float:
            return float(torch.as_tensor(self.get_parameter_value(key, required=True)).detach().reshape(-1)[0])

        def scalar(key: str) -> Tensor:
            return torch.as_tensor(number(key), dtype=dtype, device=device)

        height = scalar("hydrosphere.evapotranspiration.reference_height")
        displacement = scalar("hydrosphere.evapotranspiration.displacement_height")
        momentum_roughness = scalar("hydrosphere.evapotranspiration.roughness_length")
        heat_roughness = momentum_roughness * scalar("hydrosphere.physical.heat_to_momentum_roughness_ratio")
        stable_limit = scalar("atmosphere.most.stable_correction_limit")
        unstable_limit = scalar("atmosphere.most.unstable_correction_limit")
        bounds = self.get_parameter_value("hydrosphere.numerical.aerodynamic_conductance_bounds", required=True)
        lower, upper = float(torch.as_tensor(bounds[0])), float(torch.as_tensor(bounds[1]))
        wind = torch.clamp(wind_speed, min=number("biosphere.leaf_energy_balance.minimum_wind_speed"))
        heat_capacity = density * self.specific_heat_air

        def profile(stability: Tensor) -> Tuple[Tensor, Tensor]:
            friction = friction_velocity(wind, height, momentum_roughness, displacement, stability)
            if self.free_convection == "beljaars":
                friction = gusty_friction(stability, friction)
            resistance = aerodynamic_resistance(friction, height, heat_roughness, stability, displacement)
            return friction, torch.clamp(1.0 / resistance, min=lower, max=upper)

        if self.free_convection == "beljaars":
            beta = scalar("biosphere.leaf_energy_balance.convective_velocity_coefficient")
            boundary_layer = scalar("biosphere.leaf_energy_balance.convective_boundary_layer_height")
            karman = torch.as_tensor(self.von_karman_constant, dtype=dtype, device=device)

            def convective_ratio(stability: Tensor) -> Tensor:
                # References: CTSM Technical Note, Eqs. 2.5.28-2.5.29 (U_c = beta w* for zeta < 0,
                #   w* = (-g u* theta_v* z_i / theta_v)^(1/3), z_i = 1000 m, beta = 1); Monin-Obukhov
                #   length L = -rho c_p T u*^3 / (k g H) of the owner.
                # Mathematical expression: with H / (rho c_p) = -u* theta* and zeta = (z - d) / L,
                #   w*^3 = (g / T) (H / (rho c_p)) z_i = -u*^3 zeta z_i / (k (z - d)), so
                #   (w* / u*)^2 = (-zeta z_i / (k (z - d)))^(2/3) for zeta < 0 and 0 for zeta >= 0.
                # Parameter meanings: zeta the trial stability; z_i the convective boundary-layer
                #   height; k von Karman; z - d the effective height of the profile.
                # Units: zeta, k, ratio [1]; z_i, z, d [m].
                # Domain/constraints: the 2/3 power is evaluated only on zeta < 0 (no infinite
                #   derivative at zeta = 0); continuous at zeta = 0 (ratio 0, H = 0).
                # Sign convention: unstable (zeta < 0, H > 0) only.
                unstable = stability < 0.0
                safe = torch.where(unstable, -stability, torch.ones_like(stability))
                ratio = torch.pow(safe * boundary_layer / (karman * (height - displacement)), 2.0 / 3.0)
                return torch.where(unstable, ratio, torch.zeros_like(ratio))

            def gusty_friction(stability: Tensor, still_air_friction: Tensor) -> Tensor:
                # References: CTSM Technical Note, Eqs. 2.5.24 and 2.5.28; the owner's
                #   friction_velocity u* = k U / f_m(zeta).
                # Mathematical expression: U^2 = u^2 + beta^2 w*^2 and u* = k U / f_m give
                #   u*^2 (f_m^2 / k^2 - beta^2 (w*/u*)^2) = u^2, so
                #   u* = u / sqrt(f_m^2 / k^2 - beta^2 (w*/u*)^2), with f_m / k = u / u*_0 from
                #   the owner's u*_0 = friction_velocity(u, zeta) (no second profile law).
                # Parameter meanings: u the floored wind; u*_0 the friction velocity without
                #   gustiness; beta the convective velocity coefficient.
                # Units: u, u*, U [m s-1]; f_m, beta [1].
                # Domain/constraints: where f_m^2 / k^2 <= beta^2 (w*/u*)^2 the trial zeta is too
                #   unstable to be consistent with u (U unbounded): the bracket term is floored at
                #   1e-6 (f_m / k)^2, u* is then large, g_a reaches its upper bound and Z(zeta) is
                #   near neutral, so F = zeta - Z < 0 there and the root lies above; at the root
                #   w* = beta^-1 U_c equals (g H z_i / (rho c_p T))^(1/3) of the balance.
                # Sign convention: positive magnitudes.
                inverse = wind / still_air_friction  # f_m / k
                bracket = inverse * inverse - beta * beta * convective_ratio(stability)
                bracket = torch.clamp(bracket, min=1.0e-6 * inverse * inverse)
                return wind / torch.sqrt(bracket)

        def target(stability: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            friction, conductance = profile(stability)
            difference = leaf_air_difference(conductance)
            length = monin_obukhov_length(friction, density, self.specific_heat_air, air_temperature,
                                          heat_capacity * conductance * difference)
            mapped = torch.clamp(most_zeta(height.expand_as(length), length, displacement.expand_as(length)),
                                 min=-unstable_limit, max=stable_limit)
            return mapped, conductance, difference

        del neutral_conductance  # the neutral g_a is the zeta = 0 point of profile(); no separate start
        low = (-unstable_limit).expand_as(air_temperature).clone()
        high = stable_limit.expand_as(air_temperature).clone()
        with torch.no_grad():
            for _ in range(self.stability_iterations):
                middle = 0.5 * (low + high)
                mapped, _, _ = target(middle)
                below = middle < mapped  # F(middle) < 0: the root lies above the middle
                low = torch.where(below, middle, low)
                high = torch.where(below, high, middle)
        middle = (0.5 * (low + high)).detach()
        with torch.no_grad():
            # Slope of Z at the root by a central difference (no gradient); F' = 1 - Z'.
            step = 1.0e-4
            slope = (target(torch.clamp(middle + step, max=stable_limit))[0]
                     - target(torch.clamp(middle - step, min=-unstable_limit))[0]) / (2.0 * step)
            denominator = torch.clamp(1.0 - slope, min=1.0e-3)
        mapped, _, bracket_difference = target(middle)
        # Implicit-function step: zeta = zeta* + (Z(zeta*) - zeta*) / (1 - Z'(zeta*)); its value is
        #   zeta* up to the bracket width and its gradient is dZ/dtheta / (1 - dZ/dzeta), the
        #   derivative of the root (1 - Z' is floored at 1e-3 where the root is degenerate).
        stability = torch.clamp(middle + (mapped - middle) / denominator, min=-unstable_limit, max=stable_limit)
        friction, conductance = profile(stability)
        difference = leaf_air_difference(conductance)
        gust: Dict[str, Tensor] = {}
        if self.free_convection == "beljaars":
            # U_c = beta (w*/u*) u* and U = sqrt(u^2 + U_c^2) at the returned stability.
            convective = beta * torch.sqrt(convective_ratio(stability)) * friction
            gust = {"convective_velocity": convective,
                    "gusty_wind_speed": torch.sqrt(wind * wind + convective * convective)}
        return difference, conductance, stability, (difference - bracket_difference.detach()).abs(), gust

__all__ = ["LeafEnergyBalance"]
