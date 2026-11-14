"""
Ozone Stress Response Model for RELIC

Models plant responses to ozone exposure using stomatal flux-based
approaches and dose-response relationships.

References
----------
- Fuhrer, J. et al. (1997). Critical levels for ozone effects on vegetation
  in Europe. Environmental Pollution 97, 91-106
- Mills, G. et al. (2011). A synthesis of AOT40-based response functions and
  critical levels of ozone for agricultural and horticultural crops.
  Atmospheric Environment 45, 1703-1713
- Emberson, L.D. et al. (2000). Modelling stomatal ozone flux across Europe.
  Environmental Pollution 109, 403-413
- Sitch, S. et al. (2007). Indirect radiative forcing of climate change through
  ozone effects on the land-carbon sink. Nature 448, 791-794
- Leung, F. et al. (2020). Calibrating soybean parameters in JULES 5.0 from the
  US-Ne2/3 FLUXNET sites and the SoyFACE-O3 experiment. Geosci. Model Dev. 13,
  6201-6213, Eq. (5)

Mathematical Framework
----------------------
1. Ozone flux: F_O3 = (g_s / k_O3) * [O3] * uptake_efficiency
2. POD (Phytotoxic Ozone Dose): POD = sum(max(0, F_O3 - F_threshold) * dt)
3. AOT40 (Accumulated exposure Over Threshold): AOT40 = sum(max(0, [O3] - 40ppb) * dt)
4. Stress response: f_O3 = 1 / (1 + (POD/POD_crit)^m)

Units
-----
- Ozone concentration: mol mol^-1 (mole fraction, no conversion is applied;
  40 ppb = 4.0e-8 mol mol^-1)
- Stomatal conductance g_s: mol m-2 s-1, leaf-level conductance to water vapour
  per unit leaf area; supplied as forcing or state, or linked from its one
  tendency provider medlyn_stomatal_conductance (contract stomatal_conductance),
  which writes the same leaf-area basis and unit, so no conversion applies
- k_O3: dimensionless ratio of the leaf resistance for O3 to that for water
  vapour (Sitch et al. 2007), from the physical-constant SSOT
- Stomatal flux: mol O3 m-2 s-1
- AOT40: ppb h
- POD: mmol O3 m-2
- Stress factors: dimensionless [0-1]

Temporal support
----------------
The law is an instantaneous flux response with no internal history and no dt
limit. In a daily graph (dt = 86400 s) it reads the daily-mean [O3] and the
daily-mean g_s of its provider; this is a declared limitation, not a daily law:
the daily mean of g_s [O3] differs from the product of the daily means by their
diurnal covariance, the clipped response is convex in F_O3 (so f_O3 of the mean
flux is not the mean of f_O3), and C_detox, the flux threshold Y of POD_Y
(Mills et al. 2011) and the 40 nmol mol-1 AOT40 threshold are defined on hourly
values. Y (flux_threshold) is declared but not read by this law.
"""

from typing import Dict, Optional, Any
import torch
from torch import Tensor

from relic.core.functor import BiologicalProcessFunctor
from relic.core.abstractions import StateDict, FluxDict
from relic.core.functor import (
    DependencyDeclaration,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
)
from relic.computation.operators.iphs import advance_reaction_network
from relic.core.state import UpdateItem, UpdateMode
from relic.utils.physical_constants import get_physical_constant
from relic.scientific.atmosphere import daylight_quadrature

# Physical-constant SSOT key of k_O3, the ratio of the leaf resistance for O3 to that
# for water vapour (Sitch et al. 2007).
OZONE_RESISTANCE_RATIO_KEY = "relic.physical.constants.conversions.h2o_o3_diffusivity_ratio"


class OzoneStress(BiologicalProcessFunctor):
    """
    Ozone Stress Response Model for RELIC.
    
    Implements ozone stress effects using stomatal flux-based approaches
    and critical dose thresholds. Includes both AOT40 and POD calculations.
    
    Mathematical Framework
    ----------------------
    Ozone flux: F_O3 = (g_s / k_O3) * [O3] * uptake_efficiency
    Stress factor: f_O3 = 1 / (1 + (POD/POD_crit)^m)
    
    where:
    - g_s: leaf-level stomatal conductance to water vapour (mol m-2 s-1),
      forcing or the output of medlyn_stomatal_conductance
    - k_O3: ratio of the leaf resistance for O3 to that for water vapour (1)
    - [O3]: ozone concentration (mol mol^-1)
    - POD: Phytotoxic Ozone Dose (mmol O3 m-2)
    
    References
    ----------
    - Fuhrer, J. et al. (1997). Critical levels for ozone effects on vegetation
      in Europe. Environmental Pollution 97, 91-106
    - Mills, G. et al. (2011). A synthesis of AOT40-based response functions.
      Atmospheric Environment 45, 1703-1713
    - Emberson, L.D. et al. (2000). Modelling stomatal ozone flux across Europe.
      Environmental Pollution 109, 403-413
    - Sitch, S. et al. (2007). Nature 448, 791-794
    - Leung, F. et al. (2020). Geosci. Model Dev. 13, 6201-6213, Eq. (5)
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    # Declared temporal supports; config["temporal_support"], and an absent key keeps
    # "instantaneous" (the law once on the supplied inputs):
    #   "daylight_gauss": daily step only; the law at the daylight Gauss samples of
    #     relic.scientific.atmosphere.daylight_quadrature with the per-sample Medlyn leaf
    #     conductance stomatal_conductance_daylight_samples (replaces
    #     stomatal_conductance) and the daily [O3] held constant; outputs are 24-hour
    #     means plus ozone_stress_factor_daylight_samples for a daylight_gauss MEGAN.
    TEMPORAL_SUPPORTS = ('instantaneous', daylight_quadrature.DAYLIGHT_GAUSS)
    DAYLIGHT_INPUTS = ('latitude', 'day_of_year')
    
    def __init__(
        self,
        name: str = "ozone_stress",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        # Initialize metadata for proper Functor integration
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description=(
                'Ozone stress response model using an explicit IPHS reaction '
                'network for stomatal uptake, detoxification, and damage fluxes'
            ),
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            parameters={
                'biosphere.stress.ozone.ozone_threshold': (
                    'Ozone mixing-ratio threshold of the stress [mol mol-1]'
                ),
                'biosphere.stress.ozone.flux_threshold': (
                    'Stomatal ozone flux threshold Y of POD_Y [nmol O3 m-2 s-1]'
                ),
                'biosphere.stress.ozone.ozone_uptake_rate': (
                    'Fraction of the ozone flux taken up [dimensionless]'
                ),
                OZONE_RESISTANCE_RATIO_KEY: (
                    'Ratio k_O3 of the leaf resistance for O3 to that for water '
                    'vapour, g_s,O3 = g_s / k_O3 [dimensionless]'
                ),
                'biosphere.stress.ozone.ozone_detox_capacity': (
                    'Ozone detoxification capacity [mol O3 m-2 s-1]'
                ),
                'biosphere.stress.ozone.antioxidant_capacity': (
                    'Antioxidant capacity against ozone [mol O3 m-2]'
                ),
                'biosphere.stress.ozone.max_flux_ratio': (
                    'Upper bound of the uptake to detoxification ratio [dimensionless]'
                ),
                'biosphere.stress.global.factor_min': (
                    'Lower bound of the stress factor [dimensionless]'
                ),
                'biosphere.stress.global.factor_max': (
                    'Upper bound of the stress factor [dimensionless]'
                ),
            },
            mathematical_expression=(
                'F_O3 = eta [O3] g_s / k_O3; x_dot = N_h r_h, '
                'r_h = [F_O3, min(F_O3, C_detox), max(F_O3 - C_detox, 0)]'
            ),
            references=[
                'Fuhrer, J. et al. (1997). Environmental Pollution 97, 91-106',
                'Mills, G. et al. (2011). Atmospheric Environment 45, 1703-1713',
                'Emberson, L.D. et al. (2000). Environmental Pollution 109, 403-413',
                'Sitch, S. et al. (2007). Nature 448, 791-794',
                'Leung, F. et al. (2020). Geosci. Model Dev. 13, 6201-6213, Eq. (5)',
            ],
            input_variables=[
                'o3_mixing_ratio',
                'stomatal_conductance',
            ],
            output_variables=[
                'ozone_stress_factor',
                'ozone_uptake',
                'antioxidant_depletion',
                'diagnostics.ozone_stress.detoxification_flux',
                'diagnostics.ozone_stress.excess_ozone_flux',
                'diagnostics.ozone_stress.antioxidant_pool_tendency',
                'diagnostics.ozone_stress.damage_tendency',
                'diagnostics.ozone_stress.port_power',
                'diagnostics.ozone_stress.entropy_production',
            ],
            # No closed ozone ledger: the atmospheric O3 mole fraction is forcing and no
            # leaf ozone stock is advanced; the network only partitions the uptake flux
            # (F = D + X) into REPLACE rate outputs.
            conservation_laws=[],
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.FLUX,
        )
        configuration = dict(config or {})
        temporal_support = configuration.get('temporal_support', 'instantaneous')
        if temporal_support not in self.TEMPORAL_SUPPORTS:
            raise ValueError(
                'OzoneStress temporal_support must be one of '
                f'{list(self.TEMPORAL_SUPPORTS)}, got {temporal_support!r}'
            )
        daylight_rule = None
        if temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            daylight_rule = daylight_quadrature.daylight_config(configuration, 'OzoneStress')
            metadata.input_variables[:] = [
                'stomatal_conductance_daylight_samples' if key == 'stomatal_conductance' else key
                for key in metadata.input_variables
            ]
            metadata.input_variables.extend(self.DAYLIGHT_INPUTS)
            metadata.output_variables.append('ozone_stress_factor_daylight_samples')
        else:
            daylight_quadrature.reject_daylight_keys(configuration, 'OzoneStress')
        
        # Initialize FunctorBase with proper metadata
        super().__init__(name, config or {}, metadata=metadata)
        self.temporal_support = temporal_support
        self._daylight_rule = daylight_rule
        # No provider is declared for stomatal_conductance: FarquharPhotosynthesis
        # evaluates Ac and Aj at the atmospheric CO2 mole fraction and outputs no
        # conductance. It is read as forcing or state, or linked explicitly in a
        # graph from medlyn_stomatal_conductance, its one contract tendency
        # writer (leaf-area basis, mol m-2 s-1). The ozone provider is the static
        # optional declaration of get_dependencies.

        self.device = device if device is not None else torch.device("cpu")
        
        # Unified namespace for all ozone stress parameters
        self.namespace = "biosphere.stress.ozone"
        
        # Note: Parameters are validated in forward() method where they are actually used
        # __init__ only sets up metadata and namespace - no need to fetch parameters here
        # This follows the pattern: configure in __init__, fetch in forward()
        
        # Note: Parameters are obtained dynamically in forward() method
        # This ensures they can be updated during calibration and respects the parameter system
        # No register_buffer needed - parameters accessed via get_parameter_value() in forward()
            
    @property
    def antioxidant_capacity(self) -> Tensor:
        """Return antioxidant capacity parameter as a tensor on the configured device."""

        value = self.get_parameter_value(
            "antioxidant_capacity",
            namespace=self.namespace, required=True)
        if isinstance(value, Tensor):
            return value.to(self.device)
        return torch.tensor(float(value), device=self.device)

    def get_dependencies(self) -> DependencyDeclaration:
        """Declare atmospheric_chemistry as the optional same-step ozone provider.

        o3_mixing_ratio [mol mol-1] is the atmospheric ozone state written by
        atmospheric_chemistry (and by eddy_diffusion_blm and
        dry_deposition_aggregator); when atmospheric_chemistry is registered it
        runs first, otherwise the committed or caller-supplied state is read and
        a missing value raises. No provider is declared for stomatal_conductance
        [mol m-2 s-1]: it is forcing or state, or an explicit graph link from
        medlyn_stomatal_conductance (its contract tendency writer). Instance
        declarations are kept.
        """
        return self._merge_declared_dependencies({"optional": ["atmospheric_chemistry"]})

    def get_provides(self) -> list:
        """Define output variables provided by this Functor (the declared outputs)."""
        return list(self.metadata.output_variables)

    def _to_tensor_like(self, value: Any, like: Tensor) -> Tensor:
        """Convert scalar or tensor values to the runtime tensor layout."""

        if isinstance(value, Tensor):
            return value.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(value, device=like.device, dtype=like.dtype)

    def _extract_ozone_inputs(self, state: Dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Extract required ozone inputs with fail-fast semantics."""

        if 'o3_mixing_ratio' not in state:
            raise KeyError(
                f"{self.name}: Required state variable 'o3_mixing_ratio' [mol mol-1] "
                "(atmospheric ozone mole fraction written by atmospheric_chemistry or "
                f"supplied as state) not found. Available keys: {list(state.keys())}"
            )
        ozone_conc = state['o3_mixing_ratio']

        if 'stomatal_conductance' not in state:
            raise KeyError(
                f"{self.name}: Required forcing 'stomatal_conductance' [mol m-2 s-1] "
                "(leaf-level stomatal conductance to water vapour per unit leaf area, "
                "supplied as forcing or state or linked from medlyn_stomatal_conductance) "
                f"not found. Available keys: {list(state.keys())}"
            )
        stomatal_cond = state['stomatal_conductance']

        if not isinstance(ozone_conc, Tensor):
            ozone_conc = torch.as_tensor(ozone_conc, device=self.device, dtype=torch.float32)
        else:
            ozone_conc = ozone_conc.to(self.device)

        if not isinstance(stomatal_cond, Tensor):
            stomatal_cond = torch.as_tensor(
                stomatal_cond,
                device=ozone_conc.device,
                dtype=ozone_conc.dtype,
            )
        else:
            stomatal_cond = stomatal_cond.to(device=ozone_conc.device, dtype=ozone_conc.dtype)

        return ozone_conc, stomatal_cond

    def _evaluate_iphs_ozone_network(
        self,
        state: Dict[str, Tensor],
        *,
        dt: float,
    ) -> FluxDict:
        """Evaluate the canonical IPHS ozone uptake/detox reaction network."""
        # References: Sitch et al. (2007) as given by Leung et al. (2020), GMD 13,
        #   6201, Eq. (5); Emberson et al. (2000); Mills et al. (2011); RELIC
        #   instantaneous closure.
        # Mathematical expression: F=eta*c_O3*g_s/k_O3; D=min(F,C_detox); X=max(F-D,0);
        #   f_O3=clip(1-min(max(F/C_detox,0),r_max),f_min,f_max).
        # Parameter meanings: g_s is the leaf-level stomatal conductance to water
        #   vapour (forcing or Medlyn output); k_O3 is the ratio of the leaf resistance
        #   for O3 to that for water vapour, so g_s/k_O3 is the O3 conductance; c_O3 is
        #   the ozone mole fraction; eta is uptake efficiency; C_detox is the
        #   detoxification flux capacity.
        # Units: g_s [mol m-2 s-1]; k_O3 [1]; c_O3 [mol mol-1];
        #   F,D,X,C_detox [mol O3 m-2 s-1].
        # Domain/constraints: This is an instantaneous flux response, not accumulated
        #   POD/AOT40. F is the R_a -> 0 limit of Sitch's [O3]/(R_a + k_O3/g_l): the
        #   supplied c_O3 is taken as the leaf-surface value. The molar form equals the
        #   m s-1 form because g_s = g_l P/(R T) and [O3] = c_O3 P/(R T).
        # Sign convention: F is positive uptake; antioxidant tendency is -D; damage
        #   tendency is +X. Diagnostic rates do not commit antioxidant or damage stocks.
        # No-canopy and zero-input limit: the only divisors are the parameters k_O3,
        #   C_detox, C_ax and the 40 nmol mol-1 threshold, never g_s or LAI; g_s = 0
        #   (Medlyn nonpositive_leaf_area "no_canopy") or c_O3 = 0 gives F = 0 and
        #   f_O3 = clip(1, f_min, f_max) = 1 with the default bounds.
        # Temporal support: the outputs are the rates x_dot = N_h r_h, independent of
        #   dt (dt only forms the network increment, which is not returned); with
        #   dt = 86400 s the inputs are daily means (module docstring, declared
        #   limitation).

        ozone_conc, stomatal_cond = self._extract_ozone_inputs(state)

        stress_min = self._to_tensor_like(
            self.get_parameter_value(
                "global.factor_min",
                namespace="biosphere.stress",
                required=True,
            ),
            ozone_conc,
        )
        stress_max = self._to_tensor_like(
            self.get_parameter_value(
                "global.factor_max",
                namespace="biosphere.stress",
                required=True,
            ),
            ozone_conc,
        )
        ozone_threshold = self._to_tensor_like(
            self.get_parameter_value(
                "ozone_threshold",
                namespace=self.namespace,
                required=True,
            ),
            ozone_conc,
        )
        ozone_uptake_rate = self._to_tensor_like(
            self.get_parameter_value(
                "ozone_uptake_rate",
                namespace=self.namespace,
                required=True,
            ),
            ozone_conc,
        )
        ozone_detox_capacity = self._to_tensor_like(
            self.get_parameter_value(
                "ozone_detox_capacity",
                namespace=self.namespace,
                required=True,
            ),
            ozone_conc,
        )
        max_flux_ratio = self._to_tensor_like(
            self.get_parameter_value(
                "max_flux_ratio",
                namespace=self.namespace,
                required=True,
            ),
            ozone_conc,
        )
        antioxidant_capacity = self._to_tensor_like(self.antioxidant_capacity, ozone_conc)
        o3_resistance_ratio = self._to_tensor_like(
            get_physical_constant(OZONE_RESISTANCE_RATIO_KEY), ozone_conc
        )

        # References:
        # - Sitch et al. (2007), leaf O3 uptake F_O3 = [O3] / (R_a + k_O3 / g_l), as
        #   given by Leung et al. (2020), GMD 13, 6201, Eq. (5).
        # - Emberson et al. (2000), stomatal ozone flux across Europe.
        # - Mills et al. (2011), flux-based ozone response functions.
        # - Grmela and Ottinger (1997), GENERIC reaction-network form.
        # Mathematical expressions:
        # - g_O3 = g_s / k_O3; F_O3 = g_O3 * [O3] * eta_uptake.
        # - r_h = [F_O3, min(F_O3, C_detox), max(F_O3 - C_detox, 0)].
        # - x_dot = N_h r_h with open-system port routing for atmospheric O3 uptake.
        # Parameter meanings:
        # - g_s: stomatal_conductance, the leaf-level stomatal conductance to water
        #   vapour per unit leaf area, forcing or the Medlyn output.
        # - k_O3: ratio of the leaf resistance for O3 to that for water vapour
        #   (physical-constant SSOT); g_O3 is the stomatal conductance to O3.
        # - [O3]: o3_mixing_ratio, the atmospheric ozone mole fraction.
        # - eta_uptake: ozone_uptake_rate, the fraction of the flux taken up.
        # - C_detox: ozone_detox_capacity.
        # - antioxidant_capacity: available antioxidant capacity.
        # Units: g_s, g_O3 [mol m-2 s-1]; k_O3, eta_uptake [dimensionless];
        #   [O3] [mol mol-1]; F_O3, C_detox [mol O3 m-2 s-1];
        #   antioxidant_capacity [mol O3 m-2].
        # Domain/constraints: no range check is applied to g_s or [O3]; the law
        #   assumes g_s >= 0 and 0 <= [O3] <= 1. R_a -> 0: the supplied mole fraction
        #   is the leaf-surface value. A canopy (ground-area) conductance is not a
        #   valid g_s.
        # Sign convention: F_O3 >= 0 is O3 uptake from the atmosphere into the leaf;
        #   the detoxification and damage rows are nonnegative partitions of it.
        stomatal_ozone_conductance = stomatal_cond / o3_resistance_ratio
        ozone_uptake_rate_flux = stomatal_ozone_conductance * ozone_conc * ozone_uptake_rate
        detoxification_flux = torch.minimum(ozone_uptake_rate_flux, ozone_detox_capacity)
        excess_ozone_flux = torch.clamp(ozone_uptake_rate_flux - detoxification_flux, min=0.0)

        species = (
            'ozone_uptake',
            'diagnostics.ozone_stress.detoxification_flux',
            'diagnostics.ozone_stress.excess_ozone_flux',
            'diagnostics.ozone_stress.antioxidant_pool_tendency',
            'diagnostics.ozone_stress.damage_tendency',
        )
        ozone_network = advance_reaction_network(
            name=f"{self.name}.ozone_detox_reaction_network",
            species=species,
            reactions=(
                'stomatal_ozone_uptake',
                'antioxidant_detoxification',
                'unquenched_ozone_damage',
            ),
            stoichiometric_N=torch.as_tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                device=ozone_conc.device,
                dtype=ozone_conc.dtype,
            ),
            rates=(ozone_uptake_rate_flux, detoxification_flux, excess_ozone_flux),
            dt=dt,
            open_system_route="port",
        )

        ozone_uptake = ozone_network.species_rate('ozone_uptake')
        flux_ratio = torch.clamp(ozone_uptake / ozone_detox_capacity, min=0.0)
        clipped_flux_ratio = torch.minimum(flux_ratio, max_flux_ratio)
        stress_factor = torch.maximum(
            torch.minimum(1.0 - clipped_flux_ratio, stress_max),
            stress_min,
        )
        # References: Ploechl et al. (2000), Planta 210, 454-467 (O3 detoxification by
        #   apoplastic ascorbate, a pool replenished from the symplast); RELIC reduced
        #   closure of the declared detoxification reaction row of N_h.
        # Mathematical expression: k_ax = D / C_ax with D = min(F, C_detox).
        # Parameter meanings: D is the detoxification flux (the reaction that consumes
        #   antioxidant, antioxidant tendency -D); C_ax is the antioxidant capacity;
        #   k_ax is the specific (per unit capacity) antioxidant consumption rate.
        # Units: D [mol O3 m-2 s-1]; C_ax [mol O3 m-2]; k_ax [s-1].
        # Domain/constraints: C_ax > 0 (parameter bounds); k_ax >= 0 is a rate, not a
        #   fraction, so no dimensionless clamp applies. The antioxidant pool is not a
        #   RELIC state: it is replenished (Ploechl et al. 2000), so no per-step
        #   consumed fraction of a non-regenerated pool is formed.
        # Sign convention: k_ax >= 0 is consumption; the excess X is ozone damage and
        #   consumes no antioxidant.
        antioxidant_depletion = detoxification_flux / antioxidant_capacity
        ozone_exposure_effort = torch.clamp(ozone_conc / ozone_threshold, min=0.0)
        port_power = ozone_uptake * ozone_exposure_effort
        entropy_production = torch.clamp(
            detoxification_flux * torch.clamp(flux_ratio, min=0.0),
            min=0.0,
        )

        # The stress factor, the uptake flux and the depletion index are instantaneous
        # values of the current inputs: each call replaces them (REPLACE, not per step);
        # the simulator would otherwise accumulate a raw tensor output without run specs.
        return {
            'ozone_stress_factor': UpdateItem(
                value=stress_factor,
                mode=UpdateMode.REPLACE,
                unit="dimensionless",
                per_timestep=False,
            ),
            'ozone_uptake': UpdateItem(
                value=ozone_uptake,
                mode=UpdateMode.REPLACE,
                unit="mol O3 m-2 s-1",
                per_timestep=False,
            ),
            'antioxidant_depletion': UpdateItem(
                value=antioxidant_depletion,
                mode=UpdateMode.REPLACE,
                unit="s-1",
                per_timestep=False,
            ),
            'diagnostics.ozone_stress.detoxification_flux': ozone_network.species_rate(
                'diagnostics.ozone_stress.detoxification_flux'
            ),
            'diagnostics.ozone_stress.excess_ozone_flux': ozone_network.species_rate(
                'diagnostics.ozone_stress.excess_ozone_flux'
            ),
            'diagnostics.ozone_stress.antioxidant_pool_tendency': ozone_network.species_rate(
                'diagnostics.ozone_stress.antioxidant_pool_tendency'
            ),
            'diagnostics.ozone_stress.damage_tendency': ozone_network.species_rate(
                'diagnostics.ozone_stress.damage_tendency'
            ),
            'diagnostics.ozone_stress.port_power': port_power,
            'diagnostics.ozone_stress.entropy_production': entropy_production,
        }
        
    def forward(self, state: StateDict, time: float, dt: float, **kwargs) -> FluxDict:
        """
        Main forward pass for ozone stress calculation.
        
        Computes ozone stress factor, uptake flux, and antioxidant depletion
        based on stomatal conductance and atmospheric ozone concentration.
        
        Parameters
        ----------
        state : StateDict
            Current system state including ozone concentration
        time : float
            Current simulation time
        dt : float
            Time step
        **kwargs
            Additional arguments
            
        Returns
        -------
        FluxDict
            Dictionary containing ozone stress outputs
        """
        # References: Emberson et al. (2000); Mills et al. (2011).
        # Mathematical expression: outputs=_evaluate_iphs_ozone_network(state,dt=dt).
        # Parameter meanings: state supplies the ozone mole fraction and the leaf
        #   stomatal conductance to water vapour (forcing or Medlyn output);
        #   all response coefficients are resolved by the canonical evaluator.
        # Units: stress factor [1]; ozone reaction outputs [mol O3 m-2 s-1].
        # Domain/constraints: Delegates once; no independent POD or AOT40 integration.
        # Sign convention: Uptake and damage are positive flux magnitudes; the evaluator
        #   owns the negative antioxidant tendency and returns proposals only.
        if self.temporal_support == daylight_quadrature.DAYLIGHT_GAUSS:
            return self._forward_daylight_gauss(state, dt)
        return self._evaluate_iphs_ozone_network(state, dt=dt)

    def _forward_daylight_gauss(self, state: StateDict, dt: float) -> FluxDict:
        """Daily means of the ozone law over the daylight Gauss samples."""
        # References: Sitch et al. (2007) as given by Leung et al. (2020), Eq. (5);
        #   Goudriaan (1986); relic.scientific.atmosphere.daylight_quadrature.
        # Mathematical expression: F_k = eta c_O3 g_s,k / k_O3; f_O3,k = clip(1 -
        #   min(max(F_k / C_detox, 0), r_max), f_min, f_max); X = sum_k W_k X_k for every
        #   output X (rates, f_O3 and the diagnostics).
        # Parameter meanings: g_s,k the Medlyn leaf conductance of sample k (night: g0);
        #   c_O3 the daily-mean ozone mole fraction held constant over the day; W_k the
        #   daylight sample weights. C_detox is compared with the flux of each instant,
        #   not with a daily-mean flux (the clipped response is convex in F).
        # Units: g_s [mol m-2 s-1]; c_O3 [mol mol-1]; F [mol O3 m-2 s-1]; f_O3 [1].
        # Domain/constraints: dt = seconds_per_day; the diurnal course of [O3] is not
        #   resolved (declared limitation: its covariance with g_s is neglected).
        # Sign convention: as the instantaneous law.
        daylight_quadrature.require_daily_step(dt, 'OzoneStress')
        node_count, shortwave_shape = self._daylight_rule
        samples_key = 'stomatal_conductance_daylight_samples'
        for key in self.metadata.input_variables:
            if key not in state:
                raise KeyError(f'{self.name}: required input {key!r} not found in state')
        conductance = state[samples_key]
        tensors = {key: torch.as_tensor(state[key]) for key in ('o3_mixing_ratio', *self.DAYLIGHT_INPUTS)}
        conductance = torch.as_tensor(conductance)
        dtype = torch.promote_types(tensors['o3_mixing_ratio'].dtype, conductance.dtype)
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        samples = daylight_quadrature.daylight_sample_state(
            tensors, {'stomatal_conductance': conductance}, node_count=node_count,
            shortwave_shape=shortwave_shape, dtype=dtype)
        sample_outputs = self._evaluate_iphs_ozone_network(samples.state, dt=dt)
        outputs = daylight_quadrature.daily_outputs(sample_outputs, samples.support)
        factor = sample_outputs['ozone_stress_factor']
        outputs['ozone_stress_factor_daylight_samples'] = UpdateItem(
            value=factor.value, mode=UpdateMode.REPLACE, unit=factor.unit, per_timestep=False)
        return outputs


__all__ = ['OzoneStress']
