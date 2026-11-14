"""Conditional Medlyn leaf conductance under an explicit canopy-air assumption."""

from typing import Any, Optional

import torch

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
from relic.data.standards.unit_converter import ConversionContext, unit_converter
from relic.utils.physical_constants import get_physical_constant
from relic.utils.unified_response_functions import beer_lambert_effective_leaf_area
from relic.scientific.atmosphere import daylight_quadrature


class MedlynStomatalConductance(BiologicalProcessFunctor):
    """BiologicalProcess diagnostic, conditional on externally supplied assimilation.

    This is not an intercellular-CO2/stomatal joint solution or full CTSM.
    Its only mode explicitly substitutes canopy air for leaf-surface conditions.
    """

    FUNCTOR_NAME = "medlyn_stomatal_conductance"
    # Explicit bulk canopy conductance output; config["canopy_conductance_upscaling"]
    # selects it, and an absent key keeps the leaf-only diagnostic unchanged:
    #   "effective_leaf_area_index": g_c = g_s L_eff R T / P, the representative leaf
    #     conductance over the same Beer-Lambert effective leaf area L_eff that inverts
    #     A_leaf = A_ground / L_eff, so canopy assimilation and conductance share one
    #     leaf support (de Pury and Farquhar 1997; Monteith and Unsworth 2013, big-leaf
    #     canopy; Campbell and Norman 1998 and Nobel 2009, g_mol = g P / (R T)).
    CANOPY_CONDUCTANCE_UPSCALINGS = ("effective_leaf_area_index",)
    # Inputs declared (and required) only with a canopy conductance output: the
    # canopy-air temperature [K] and pressure [Pa] of the molar-to-velocity conversion.
    CANOPY_CONVERSION_INPUTS = ("air_temperature", "atmospheric_pressure")
    # Declared handling of leaf_area_index <= 0; config["nonpositive_leaf_area"], and an
    # absent key keeps "raise":
    #   "raise": the representative-leaf inversion is undefined and raises ValueError.
    #   "no_canopy": no leaves exist, so no leaf or canopy conductance: g_s = 0 and
    #     g_c = 0, flagged in diagnostics.medlyn.no_canopy; needs a canopy output.
    NONPOSITIVE_LEAF_AREA_POLICIES = ("raise", "no_canopy")
    # Declared temporal supports; config["temporal_support"], and an absent key keeps
    # "instantaneous" (the law once on the supplied inputs):
    #   "daylight_gauss": daily step only; the law at the daylight Gauss samples of
    #     relic.scientific.atmosphere.daylight_quadrature with the per-sample Farquhar
    #     assimilation net_photosynthesis_daylight_samples (replaces net_photosynthesis)
    #     and the sample deficit e0(T_k) - e_a (actual_vapor_pressure replaces
    #     vapor_pressure_deficit); outputs are 24-hour means plus the per-sample leaf
    #     (and, with an upscaling, canopy) conductance for daylight_gauss readers.
    TEMPORAL_SUPPORTS = ("instantaneous", daylight_quadrature.DAYLIGHT_GAUSS)
    DAYLIGHT_INPUTS = ("air_temperature", "air_temperature_max", "air_temperature_min",
                       "latitude", "day_of_year")

    def __init__(self, name: str = FUNCTOR_NAME, config: Optional[dict] = None,
                 *, parameter_manager: Any = None, device: Any = None) -> None:
        configuration = dict(config or {})
        if configuration.get("surface_conditions") != "leaf_surface_equals_canopy_air":
            raise ValueError(
                "Medlyn surface_conditions must explicitly declare "
                "'leaf_surface_equals_canopy_air'; measured leaf-surface mode is not implemented."
            )
        upscaling = configuration.get("canopy_conductance_upscaling")
        if upscaling is not None and upscaling not in self.CANOPY_CONDUCTANCE_UPSCALINGS:
            raise ValueError(
                "Medlyn canopy_conductance_upscaling must be absent or one of "
                f"{list(self.CANOPY_CONDUCTANCE_UPSCALINGS)}, got {upscaling!r}"
            )
        nonpositive_leaf_area = configuration.get("nonpositive_leaf_area", "raise")
        if nonpositive_leaf_area not in self.NONPOSITIVE_LEAF_AREA_POLICIES:
            raise ValueError(
                "Medlyn nonpositive_leaf_area must be one of "
                f"{list(self.NONPOSITIVE_LEAF_AREA_POLICIES)}, got {nonpositive_leaf_area!r}"
            )
        if nonpositive_leaf_area == "no_canopy" and upscaling is None:
            raise ValueError(
                "Medlyn nonpositive_leaf_area 'no_canopy' declares a zero canopy conductance "
                "and needs canopy_conductance_upscaling"
            )
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=("Conditional leaf conductance: leaf_surface_equals_canopy_air; "
                         "ground assimilation inverted with Farquhar L_eff; explicit residual branches."),
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression=(
                "L_eff=(1-exp(-k*LAI))/k; A_leaf=A_ground/L_eff; "
                "D_eff=max(D_air,D_min); "
                "g_s=g0+r_wc*(1+g1/sqrt(D_eff_kPa))*A_leaf/C_air_ppm "
                "if ppfd>0 and A_leaf>0, else g0"
            ),
            references=[
                "Medlyn et al. (2011), doi:10.1111/j.1365-2486.2010.02375.x.",
                "CTSM Photosynthesis technical note, Eq. 2.9.1.",
                "CTSM PhotosynthesisMod.F90, commit 7d1bc988a5a67660d2988c57a0bf195173c31017.",
                "de Pury and Farquhar (1997), Plant Cell Environ. 20, 537-557.",
            ],
            parameters={
                "coupling.scales.biosphere_atmosphere.g0": "Residual leaf water conductance [mol m-2 s-1]",
                "coupling.scales.biosphere_atmosphere.g1": "Medlyn slope prior [kPa^0.5]",
                "coupling.scales.biosphere_atmosphere.vpd_min_pa": "Declared VPD regularization floor [Pa]",
                "biosphere.photosynthesis.canopy_light_extinction": "Farquhar canopy extinction [dimensionless]",
                "relic.physical.constants.conversions.h2o_co2_diffusivity_ratio": "Water to CO2 diffusivity ratio [dimensionless]",
            },
            input_variables=["net_photosynthesis", "leaf_area_index", "ppfd",
                             "atmospheric_co2_ppm", "vapor_pressure_deficit"],
            output_variables=["stomatal_conductance", "diagnostics.medlyn.leaf_net_assimilation",
                              "diagnostics.medlyn.dark_residual",
                              "diagnostics.medlyn.nonpositive_assimilation_residual",
                              "diagnostics.medlyn.vpd_floor_active"],
            conservation_laws=[],
            spatial_dimensions=0,
            spatial_scale=SpatialScale.PATCH,
            temporal_scale=TemporalScale.HOUR,
            variable_kind=VariableKind.DIAGNOSTIC,
            conflict_resolution_policy=ConflictResolutionPolicy.EXCLUSIVE,
        )
        if upscaling is not None:
            metadata.input_variables.extend(self.CANOPY_CONVERSION_INPUTS)
            metadata.output_variables.append("canopy_conductance")
        if nonpositive_leaf_area == "no_canopy":
            metadata.output_variables.append("diagnostics.medlyn.no_canopy")
        temporal_support = configuration.get("temporal_support", "instantaneous")
        if temporal_support not in self.TEMPORAL_SUPPORTS:
            raise ValueError(
                "Medlyn temporal_support must be one of "
                f"{list(self.TEMPORAL_SUPPORTS)}, got {temporal_support!r}"
            )
        # The inputs of the instantaneous law, in the order it reads them.
        law_inputs = list(metadata.input_variables)
        daylight_rule = None
        if temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            daylight_rule = daylight_quadrature.daylight_config(configuration, "Medlyn")
            replaced = {"net_photosynthesis": "net_photosynthesis_daylight_samples",
                        "vapor_pressure_deficit": "actual_vapor_pressure"}
            metadata.input_variables[:] = [replaced.get(key, key) for key in metadata.input_variables]
            metadata.input_variables.extend(
                key for key in self.DAYLIGHT_INPUTS if key not in metadata.input_variables)
            metadata.output_variables.append("stomatal_conductance_daylight_samples")
            if upscaling is not None:
                metadata.output_variables.append("canopy_conductance_daylight_samples")
        else:
            daylight_quadrature.reject_daylight_keys(configuration, "Medlyn")
        super().__init__(name=name, config=configuration, metadata=metadata,
                         parameter_manager=parameter_manager, device=device)
        self.canopy_conductance_upscaling = upscaling
        self.nonpositive_leaf_area = nonpositive_leaf_area
        self.temporal_support = temporal_support
        self._law_inputs = law_inputs
        self._daylight_rule = daylight_rule

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        # References: Goudriaan (1986), Agric. For. Meteorol. 38, 249-254;
        #   relic.scientific.atmosphere.daylight_quadrature (daylight_gauss support).
        # Mathematical expression: instantaneous support: the law below on the inputs;
        #   daylight_gauss: g_s = sum_k W_k g_s(A_k, I_k, D_k, T_k), A_k the Farquhar
        #   net assimilation of the same sample (net_photosynthesis_daylight_samples).
        # Parameter meanings: W_k the daylight sample weights; I_k, D_k, T_k the sample
        #   ppfd (dark selector), vapour pressure deficit and air temperature.
        # Units: as the law; samples on a trailing axis of length n + 1.
        # Domain/constraints: daylight_gauss requires dt = seconds_per_day.
        # Sign convention: conductances are nonnegative magnitudes.
        if self.temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            return self._forward_daylight_gauss(state, dt)
        return self._instantaneous_law(state)

    def _forward_daylight_gauss(self, state: StateDict, dt: float) -> FluxDict:
        """Daily means of the Medlyn law over the daylight Gauss samples."""
        # References: Medlyn et al. (2011); Goudriaan (1986); daylight_quadrature.
        # Mathematical expression: g_s,k = law(A_k, LAI, I_k, C_a, D_k[, T_k, P]);
        #   g_s = sum_k W_k g_s,k; g_c = sum_k W_k g_c,k (daytime g_c times N/24 plus the
        #   night conductance times 1 - N/24); D_k = max(e0(T_k) - e_a, 0).
        # Parameter meanings: A_k the sample net assimilation of the daylight_gauss
        #   Farquhar node (night: -R_d, which selects the g0 branch); I_k the 24-hour mean
        #   ppfd redistributed by the solar elevation (I_night = 0: dark branch g0).
        # Units: g_s [mol m-2 s-1]; g_c [m s-1]; A [umol CO2 m-2 s-1].
        # Domain/constraints: dt = seconds_per_day; the input checks of the law apply to
        #   every sample.
        # Sign convention: conductances are nonnegative magnitudes.
        daylight_quadrature.require_daily_step(dt, "Medlyn")
        node_count, shortwave_shape = self._daylight_rule
        samples_key = "net_photosynthesis_daylight_samples"
        ground = self._ensure_tensor(state[samples_key])
        if not ground.is_floating_point():
            raise TypeError(f"{samples_key} must use a floating tensor dtype.")
        values = {key: self._ensure_tensor(state[key], dtype=ground.dtype)
                  for key in self.metadata.input_variables if key != samples_key}
        samples = daylight_quadrature.daylight_sample_state(
            values, {"net_photosynthesis": ground}, node_count=node_count,
            shortwave_shape=shortwave_shape, flux_keys=("ppfd",), temperature=True,
            vapor_pressure_deficit=True, dtype=ground.dtype)
        sample_outputs = self._instantaneous_law(samples.state)
        outputs = daylight_quadrature.daily_outputs(sample_outputs, samples.support)
        for key in ("stomatal_conductance", "canopy_conductance"):
            if key in sample_outputs:
                outputs[key + "_daylight_samples"] = UpdateItem(
                    value=sample_outputs[key].value, mode=UpdateMode.REPLACE,
                    unit=sample_outputs[key].unit, per_timestep=False)
        return outputs

    def _instantaneous_law(self, state: StateDict) -> FluxDict:
        # References: de Pury and Farquhar (1997), Plant Cell Environ. 20, 537-557.
        # Mathematical expression: L_eff=(1-exp(-k*LAI))/k; A_leaf=A_ground/L_eff.
        # Parameter meanings: k is the shared Farquhar extinction coefficient;
        #   LAI is total leaf area per ground area; L_eff is illuminated leaf support.
        # Units: k [1]; LAI,L_eff [m2 leaf m-2 ground]; A_ground [umol CO2 m-2 ground s-1];
        #   A_leaf [umol CO2 m-2 leaf s-1].
        # Domain/constraints: all inputs finite; non-scalar input shapes must agree;
        #   scalar inputs broadcast to this common support, never across patch axes.
        #   LAI>0 and L_eff numerically positive;
        #   C_air_ppm>0. Canopy-air substitution is explicit at construction.
        # Sign convention: positive assimilation is CO2 uptake; signed A is preserved.
        ground = self._ensure_tensor(state["net_photosynthesis"])
        if not ground.is_floating_point():
            raise TypeError("net_photosynthesis must use a floating tensor dtype.")
        values = {"net_photosynthesis": ground}
        for key in self._law_inputs:
            value = self._ensure_tensor(state[key], dtype=ground.dtype)
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{key} must be finite.")
            values[key] = value
        support_shapes = {tuple(value.shape) for value in values.values() if value.ndim}
        if len(support_shapes) > 1:
            raise ValueError("Medlyn non-scalar input shapes must share one spatial support.")
        values = dict(zip(values, torch.broadcast_tensors(*values.values()), strict=True))
        ground = values["net_photosynthesis"]
        leaf_area_index = values["leaf_area_index"]
        # References: none beyond the inversion above; an explicit configuration choice.
        # Mathematical expression: N = (LAI <= 0); LAI_inv = 1 where N else LAI.
        # Parameter meanings: N the no-canopy mask of nonpositive_leaf_area "no_canopy";
        #   LAI_inv the leaf area of the inversion, whose placeholder 1 only keeps the
        #   arithmetic finite where N holds; every result there is replaced by 0 below.
        # Units: LAI, LAI_inv [m2 leaf m-2 ground]; N [1].
        # Domain/constraints: with the default "raise" N is never formed and LAI <= 0 raises.
        # Sign convention: not applicable (mask and areas).
        if self.nonpositive_leaf_area == "no_canopy":
            no_canopy = leaf_area_index <= 0
            leaf_area_index = torch.where(no_canopy, torch.ones_like(leaf_area_index), leaf_area_index)
        if bool(torch.any(leaf_area_index <= 0)):
            raise ValueError("leaf_area_index must be positive for representative-leaf inversion.")
        if bool(torch.any(values["atmospheric_co2_ppm"] <= 0)):
            raise ValueError("atmospheric_co2_ppm must be positive.")
        extinction = self.get_parameter_value(
            "biosphere.photosynthesis.canopy_light_extinction", required=True)
        effective_lai = beer_lambert_effective_leaf_area(leaf_area_index, extinction)
        if bool(torch.any(effective_lai <= 0)):
            raise ValueError("leaf_area_index gives unresolved zero effective leaf area.")
        leaf = ground / effective_lai
        if not bool(torch.isfinite(leaf).all()):
            raise ValueError("Representative leaf assimilation overflowed the input precision.")
        if self.nonpositive_leaf_area == "no_canopy":
            leaf = torch.where(no_canopy, torch.zeros_like(leaf), leaf)

        # References: Medlyn et al. (2011), doi:10.1111/j.1365-2486.2010.02375.x;
        #   CTSM technical note Eq. 2.9.1; fixed PhotosynthesisMod.F90 commit
        #   7d1bc988a5a67660d2988c57a0bf195173c31017, lines 87-88, 1856-1934.
        # Mathematical expression: D_eff=max(D_air,D_min);
        #   g_s=g0+r_wc*(1+g1/sqrt(D_eff_kPa))*A_leaf/C_air_ppm
        #   for ppfd>0 and A_leaf>0; otherwise g_s=g0.
        # Parameter meanings: g0 is residual conductance; g1 is the Medlyn slope;
        #   r_wc is the water/CO2 diffusivity ratio; D_min is the declared VPD floor;
        #   D_air and C_air are canopy-air values assumed equal to leaf-surface values;
        #   ppfd only selects illumination; A_leaf is recovered net leaf assimilation.
        # Units: g_s,g0 [mol H2O m-2 leaf s-1]; g1 [kPa^0.5]; D_air,D_min [Pa];
        #   D_eff_kPa [kPa]; r_wc [1]; C_air_ppm [umol CO2 mol-1 air];
        #   A_leaf [umol CO2 m-2 leaf s-1]; ppfd [umol photons m-2 ground s-1].
        # Domain/constraints: declared dark/nonpositive-A residual branches and VPD
        #   regularization are implementation choices, not unmodified full-day Medlyn;
        #   no maximum-conductance cap, condensation or Ci-gs joint solution is implied.
        # Sign convention: conductance is nonnegative; positive A raises conductance.
        #   This instantaneous diagnostic replaces g_s and advances no material pool.
        g0 = self._ensure_tensor(self.get_parameter_value(
            "coupling.scales.biosphere_atmosphere.g0", units="mol m-2 s-1", required=True), dtype=ground.dtype)
        g1 = self._ensure_tensor(self.get_parameter_value(
            "coupling.scales.biosphere_atmosphere.g1", units="kPa^0.5", required=True), dtype=ground.dtype)
        vpd_min = self._ensure_tensor(self.get_parameter_value(
            "coupling.scales.biosphere_atmosphere.vpd_min_pa", units="Pa", required=True), dtype=ground.dtype)
        ratio = get_physical_constant("relic.physical.constants.conversions.h2o_co2_diffusivity_ratio")
        vpd = values["vapor_pressure_deficit"]
        effective_vpd = unit_converter.convert(
            torch.maximum(vpd, vpd_min), "vapor_pressure_deficit", "Pa", "kPa")
        dark = values["ppfd"] <= 0
        nonpositive = (~dark) & (leaf <= 0)
        active = (~dark) & (leaf > 0)
        # Evaluate only the selected assimilation branch; this arithmetic placeholder
        # does not replace signed leaf assimilation in state or diagnostics. Divide
        # by C before multiplying to avoid a needless intermediate overflow.
        active_leaf = torch.where(active, leaf, torch.zeros_like(leaf))
        daylight_conductance = g0 + (active_leaf / values["atmospheric_co2_ppm"]) * ratio * (1 + g1 / torch.sqrt(effective_vpd))
        conductance = torch.where(active, daylight_conductance, g0)
        if self.nonpositive_leaf_area == "no_canopy":
            conductance = torch.where(no_canopy, torch.zeros_like(conductance), conductance)
        if not bool(torch.isfinite(conductance).all()):
            raise ValueError("Medlyn stomatal_conductance overflowed the input precision.")
        outputs = {
            "stomatal_conductance": UpdateItem(value=conductance, mode=UpdateMode.REPLACE,
                                               unit="mol m-2 s-1", per_timestep=False),
            "diagnostics.medlyn.leaf_net_assimilation": leaf,
            "diagnostics.medlyn.dark_residual": dark,
            "diagnostics.medlyn.nonpositive_assimilation_residual": nonpositive,
            "diagnostics.medlyn.vpd_floor_active": vpd < vpd_min,
        }
        if self.canopy_conductance_upscaling is not None:
            # References: Monteith and Unsworth (2013), Principles of Environmental
            #   Physics, 4th ed. (big-leaf canopy of parallel leaf conductances);
            #   Campbell and Norman (1998), An Introduction to Environmental
            #   Biophysics, 2nd ed.; Nobel (2009), Physicochemical and Environmental
            #   Plant Physiology, 4th ed. (g_mol = g P / (R T)); UnitConverter
            #   conductance_mol_to_ms (Gerlein-Safdi et al. 2018, Eq. 13).
            #   de Pury and Farquhar (1997), Plant Cell Environ. 20, 537-557 (L_eff).
            # Mathematical expression: G_c = g_s L_eff (mol m-2 ground s-1) with
            #   L_eff = (1 - exp(-k LAI)) / k, the support of A_leaf = A_ground / L_eff,
            #   so G_c / A_ground = g_s / A_leaf; g_c = G_c R T / P (m s-1);
            #   g_c = 0 where N (no canopy).
            # Parameter meanings: g_s the leaf conductance above; L_eff the effective leaf
            #   area of the inversion above (k biosphere.photosynthesis.canopy_light_extinction);
            #   R the universal gas constant of the physical-constant SSOT; T the air
            #   temperature and P the atmospheric pressure of the canopy air
            #   (leaf_surface_equals_canopy_air).
            # Units: g_s [mol H2O m-2 leaf s-1]; L_eff, LAI [m2 leaf m-2 ground]; k [1];
            #   G_c [mol H2O m-2 ground s-1]; R [J mol-1 K-1]; T [K]; P [Pa]; g_c [m s-1].
            # Domain/constraints: T > 0 and P > 0 finite (UnitConverter raises otherwise);
            #   G_c >= 0; 0 < L_eff < min(LAI, 1 / k). No separate shaded-leaf, boundary-layer
            #   or soil-surface conductance enters.
            # Sign convention: conductances are nonnegative magnitudes.
            canopy_molar = conductance * effective_lai
            if self.nonpositive_leaf_area == "no_canopy":
                canopy_molar = torch.where(no_canopy, torch.zeros_like(canopy_molar), canopy_molar)
            canopy = unit_converter.convert(
                canopy_molar, "canopy_conductance", "mol m-2 s-1", "m s-1",
                context=ConversionContext(temperature=values["air_temperature"],
                                          pressure=values["atmospheric_pressure"]))
            outputs["canopy_conductance"] = UpdateItem(value=canopy, mode=UpdateMode.REPLACE,
                                                       unit="m s-1", per_timestep=False)
        if self.nonpositive_leaf_area == "no_canopy":
            outputs["diagnostics.medlyn.no_canopy"] = no_canopy
        return outputs
