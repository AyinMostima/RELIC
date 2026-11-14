"""
Thermal Stress Response Model for RELIC

Models plant responses to temperature extremes using Gaussian temperature
response functions and threshold-based heat/cold stress calculations.

References
----------
- Berry, J. & Bjorkman, O. (1980). Photosynthetic response and adaptation to 
  temperature in higher plants. Annual Review of Plant Physiology 31, 491-543
- Salvucci, M.E. & Crafts-Brandner, S.J. (2004). Inhibition of photosynthesis 
  by heat stress. Plant, Cell & Environment 27, 1063-1071
- Hatfield, J.L. & Prueger, J.H. (2015). Temperature extremes: Effect on plant
  growth and development. Weather and Climate Extremes 10, 4-10

Mathematical Framework
----------------------
1. Gaussian temperature response: f_temp = exp(-0.5 * ((T - T_opt)/sigma)^2)
2. Heat stress: f_heat = max(0, 1 + k_heat * (T - T_heat)) above T_heat, 0 above T_crit
3. Cold stress: f_cold = max(0, 1 + k_cold * (T_cold - T)) below T_cold, 0 below T_freeze
The three factors are returned separately, each clipped to the shared bounds.
T is the declared input air_temperature (near-surface air temperature, K), or
the daily extremum air_temperature_max or air_temperature_min selected by the
instance config key air_temperature_source, which may also name the leaf
temperature leaf_temperature or its daily-maximum estimate leaf_temperature_max
of the leaf energy balance.

Units
-----
- Temperature: K
- Stress factors: dimensionless [0-1]
- Temperature tolerance width (sigma): K
"""

from typing import Dict, Optional, Any
import torch

from relic.core.functor import BiologicalProcessFunctor
from relic.core.abstractions import StateDict, FluxDict
from relic.core.state import UpdateItem, UpdateMode
from relic.core.functor import (
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
    TemporalScale,
)


class ThermalStress(BiologicalProcessFunctor):
    """
    Thermal Stress Response Model for RELIC.
    
    Implements temperature stress responses using Gaussian temperature response
    functions combined with threshold-based heat and cold stress calculations.
    Based on Berry & Bjorkman (1980) and Salvucci & Crafts-Brandner (2004).
    
    Mathematical Framework
    ----------------------
    Gaussian response: f_temp = exp(-0.5 * ((T - T_opt)/sigma)^2)
    where:
    - T: current temperature (K)
    - T_opt: optimal temperature (K)
    - sigma: temperature tolerance width (K)
    
    References
    ----------
    - Berry, J. & Bjorkman, O. (1980). Photosynthetic response and adaptation to 
      temperature in higher plants. Annual Review of Plant Physiology 31, 491-543
    - Salvucci, M.E. & Crafts-Brandner, S.J. (2004). Inhibition of photosynthesis 
      by heat stress. Plant, Cell & Environment 27, 1063-1071
    - Hatfield, J.L. & Prueger, J.H. (2015). Temperature extremes: Effect on plant
      growth and development. Weather and Climate Extremes 10, 4-10
    """
    PHASE8_RUNTIME_REPLACEMENT_STATUS = 'canonical_iphs_runtime'
    # Selectable temperature input T of the three responses (all in K): the
    # near-surface air temperature, or its daily maximum or minimum (CF-1.8
    # section 7.3 cell_methods "time: maximum" / "time: minimum"; SCM contracts
    # air_temperature_max and air_temperature_min).
    AIR_TEMPERATURE_SOURCES = ('air_temperature', 'air_temperature_max', 'air_temperature_min')
    # Leaf temperatures of the leaf energy balance (relic.scientific.biosphere.
    # photosynthesis.leaf_energy_balance, K) that air_temperature_source may also name:
    # leaf_temperature (the balance of the step) and leaf_temperature_max (its declared
    # daily-maximum estimate T_max + dT on the daily support). The heat threshold of
    # Salvucci and Crafts-Brandner (2004) is a leaf-temperature response of Rubisco
    # activase; the cold and Gaussian parameters stay as declared for air temperature.
    LEAF_TEMPERATURE_SOURCES = ('leaf_temperature', 'leaf_temperature_max')
    # The three responses in their declaration order (all dimensionless [1]):
    # f_T (Gaussian), f_hot (heat threshold), f_cold (cold threshold).
    STRESS_OUTPUTS = ('temperature_stress_factor', 'heat_stress_factor', 'cold_stress_factor')
    
    def __init__(
        self,
        name: str = "thermal_stress",
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None
    ):
        # References: SCM variable contracts of air_temperature,
        #   air_temperature_max and air_temperature_min; the flat StateManager
        #   keys of architecture_overview.md (a graph cannot alias a state key
        #   to another input name, so the selected key is the declared input).
        # Mathematical expression: T = s[k] with k = air_temperature_source;
        #   the three responses of forward are unchanged.
        # Parameter meanings: k selects which air temperature the responses
        #   read, for example the daily maximum for the heat threshold and the
        #   daily minimum for the cold threshold of two instances.
        # Units: T [K].
        # Domain/constraints: k is one of AIR_TEMPERATURE_SOURCES or
        #   LEAF_TEMPERATURE_SOURCES; the default air_temperature keeps the former
        #   interface.
        # Sign convention: not applicable; this is a wiring selection.
        air_temperature_source = (config or {}).get(
            'air_temperature_source', 'air_temperature'
        )
        temperature_sources = self.AIR_TEMPERATURE_SOURCES + self.LEAF_TEMPERATURE_SOURCES
        if air_temperature_source not in temperature_sources:
            raise ValueError(
                "air_temperature_source must be one of "
                f"{list(temperature_sources)}, got {air_temperature_source!r}"
            )
        # References: SCM variable contracts of temperature_stress_factor,
        #   heat_stress_factor and cold_stress_factor; the single-writer rule of
        #   the shared state (core/simulator/registration.py output overlap).
        # Mathematical expression: O = {y in STRESS_OUTPUTS : y in stress_outputs};
        #   every y in O keeps its response of forward, the others are not written.
        # Parameter meanings: stress_outputs selects the declared outputs, so two
        #   instances (for example heat from the daily maximum and cold from the
        #   daily minimum air temperature) do not write the same state key.
        # Units: every output [1].
        # Domain/constraints: a non-empty duplicate-free subset; the default is all
        #   three outputs in declaration order (the former interface).
        # Sign convention: not applicable; this is an interface selection.
        configured_outputs = (config or {}).get('stress_outputs')
        if configured_outputs is None:
            stress_outputs = list(self.STRESS_OUTPUTS)
        else:
            if not isinstance(configured_outputs, (list, tuple)) or not configured_outputs:
                raise TypeError("stress_outputs must be a non-empty list or tuple")
            if len(set(configured_outputs)) != len(configured_outputs):
                raise ValueError("stress_outputs must not contain duplicates")
            unsupported_outputs = [
                output for output in configured_outputs if output not in self.STRESS_OUTPUTS
            ]
            if unsupported_outputs:
                raise ValueError(
                    "stress_outputs contains unsupported values: "
                    + ", ".join(str(output) for output in unsupported_outputs)
                )
            stress_outputs = [
                output for output in self.STRESS_OUTPUTS if output in configured_outputs
            ]
        # Initialize metadata for proper Functor integration
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description='Thermal stress response model based on temperature',
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression='f_temp = exp(-0.5 * ((T - T_opt)/sigma)^2)',
            references=[
                'Berry, J. & Bjorkman, O. (1980). Annual Review of Plant Physiology 31, 491-543',
                'Salvucci, M.E. & Crafts-Brandner, S.J. (2004). Plant, Cell & Environment 27, 1063-1071',
                'Hatfield, J.L. & Prueger, J.H. (2015). Weather and Climate Extremes 10, 4-10'
            ],
            parameters={
                'optimal_temperature': {
                    'description': 'Optimal temperature for plant processes',
                    'units': 'K',
                    'namespace': 'biosphere.stress.thermal'
                },
                'heat_stress_threshold': {
                    'description': 'Temperature threshold for heat stress',
                    'units': 'K',
                    'namespace': 'biosphere.stress.thermal'
                },
                'critical_high_temperature': {
                    'description': 'Critical high temperature causing severe damage',
                    'units': 'K',
                    'namespace': 'biosphere.stress.thermal'
                },
                'cold_stress_threshold': {
                    'description': 'Temperature threshold for cold stress',
                    'units': 'K',
                    'namespace': 'biosphere.stress.thermal'
                },
                'freezing_damage_threshold': {
                    'description': 'Temperature causing freezing damage',
                    'units': 'K',
                    'namespace': 'biosphere.stress.thermal'
                },
                'temperature_tolerance_width': {
                    'description': 'Temperature tolerance width for Gaussian response',
                    'units': 'K',
                    'namespace': 'biosphere.stress.thermal'
                },
                'heat_stress_coefficient': {
                    'description': 'Coefficient for heat stress response function',
                    'units': 'K-1',
                    'namespace': 'biosphere.stress.thermal'
                },
                'cold_stress_coefficient': {
                    'description': 'Coefficient for cold stress response function',
                    'units': 'K-1',
                    'namespace': 'biosphere.stress.thermal'
                },
                'global.factor_min': {
                    'description': 'Lower bound for stress factors',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.stress'
                },
                'global.factor_max': {
                    'description': 'Upper bound for stress factors',
                    'units': 'dimensionless',
                    'namespace': 'biosphere.stress'
                }
            },
            input_variables=[
                air_temperature_source,
            ],
            output_variables=stress_outputs,
            conservation_laws=[],
            spatial_dimensions=0,
            temporal_scale=TemporalScale.HOUR,
            spatial_scale=SpatialScale.PATCH,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        
        # Initialize FunctorBase with proper metadata
        super().__init__(name, config or {}, metadata=metadata)
        self.air_temperature_source = air_temperature_source
        self.stress_outputs = tuple(stress_outputs)
        
        self.device = device or (
            torch.device("cpu")
        )
        
        # Unified namespace for all thermal stress parameters
        self.namespace = "biosphere.stress.thermal"
        
        # Get temperature parameters from ParameterManager
        # All parameters are now stored in K in the database
        _ = self.get_parameter_value(
            "optimal_temperature",
            namespace=self.namespace,
        )
        
        _ = self.get_parameter_value(
            "heat_stress_threshold",
            namespace=self.namespace,
        )
        
        _ = self.get_parameter_value(
            "critical_high_temperature",
            namespace=self.namespace,
        )
        
        _ = self.get_parameter_value(
            "cold_stress_threshold",
            namespace=self.namespace,
        )
        
        _ = self.get_parameter_value(
            "freezing_damage_threshold",
            namespace=self.namespace,
        )
        
        _ = self.get_parameter_value(
            "temperature_tolerance_width",
            namespace=self.namespace,
        )
        
        # Heat and cold stress coefficients
        _ = self.get_parameter_value(
            "heat_stress_coefficient",
            namespace=self.namespace,
        )
        
        _ = self.get_parameter_value(
            "cold_stress_coefficient",
            namespace=self.namespace,
        )
        
        # Store parameter names for dynamic access (no caching)
            
    def get_dependencies(self) -> list:
        """Declare no Functor dependency; air_temperature is an external port.

        References:
            SCM variable contract of air_temperature
            (relic/contracts/variables/atmosphere/air_temperature.yaml: kind
            forcing, owner null); architecture_overview.md Sections 4.2 and
            4.3 (dependency declaration and its single merge owner).
        Mathematical expression:
            D_static = {}; D = D_static U D_instance
            (FunctorBase._merge_declared_dependencies).
        Parameter meanings:
            D_static is this class-level declaration; D_instance holds the
            names bound by config["dependencies"], set_dependencies(...) or
            add_dependency(...), for example the Functor links of a graph; D
            is the declaration read by the dependency graph.
        Units:
            Not applicable to Functor names. The only input air_temperature
            (or the selected air_temperature_max or air_temperature_min) is
            in K; the three outputs are dimensionless [1].
        Domain/constraints:
            No relic process Functor writes air_temperature, so no provider
            is declared for it: the caller supplies it as external forcing
            or committed state, and a missing value raises in forward.
            environmental_forcing writes only vapor_pressure_deficit, which
            this Functor does not read. Instance declarations are kept. A
            leaf source (leaf_temperature, leaf_temperature_max) is written by
            leaf_energy_balance or supplied as state; its graph link orders
            the two nodes, so no static dependency is declared for it either.
        Sign convention:
            Not applicable; a dependency edge orders execution and carries
            no flux.
        """
        return self._merge_declared_dependencies([])
        
    def get_provides(self) -> list:
        """Define output variables provided by this Functor (stress_outputs)."""
        return list(self.stress_outputs)
        
    def forward(self, state: StateDict, time: float, dt: float, **kwargs) -> FluxDict:
        """
        Main forward pass for thermal stress calculation.
        
        Computes three stress components:
        1. Overall temperature stress (Gaussian response)
        2. Heat stress (threshold-based)
        3. Cold stress (threshold-based)
        
        Parameters
        ----------
        state : StateDict
            Current system state; the declared input air_temperature [K] is the
            near-surface air temperature
        time : float
            Current simulation time
        dt : float
            Time step
        **kwargs
            Additional arguments
            
        Returns
        -------
        FluxDict
            Dictionary containing all temperature stress components
        """
        # References: Berry and Bjorkman (1980); Salvucci and Crafts-Brandner (2004);
        #   Hatfield and Prueger (2015).
        # Mathematical expression: f_T=exp(-0.5*((T-T_opt)/sigma_T)^2);
        #   f_hot=max(0,1+k_hot*(T-T_hot)) above T_hot, else 1;
        #   f_cold=max(0,1+k_cold*(T_cold-T)) below T_cold, else 1.
        # Parameter meanings: k_hot/k_cold are negative slopes; hot/cold cutoffs set
        #   their factor to zero beyond critical-high/freezing temperatures.
        # Units: temperatures and width [K]; slopes [K-1]; all three outputs [1].
        # Domain/constraints: Each factor is clipped separately to shared bounds;
        #   the forward path returns three independent controls, not their product.
        # Sign convention: Smaller factors indicate stronger inhibition; outputs are
        #   instantaneous controls and are not heat fluxes or tissue-damage stocks.
        # T is the declared input air_temperature [K], the near-surface air temperature
        #   supplied as external forcing or committed state (no relic process Functor
        #   writes it): the cited thresholds are crop cardinal air
        #   temperatures (Hatfield and Prueger 2015) and the RELIC Farquhar temperature
        #   responses read the same input (big-leaf T_leaf = T_air). No alias
        #   (temperature, leaf_temperature) stands in for it; a missing input raises.
        #   With air_temperature_source the declared input is instead the daily
        #   maximum or minimum air temperature [K], read under its own state key, or
        #   the leaf temperature leaf_temperature / leaf_temperature_max [K] written by
        #   leaf_energy_balance (then the thresholds act on the big-leaf temperature).
        source = self.air_temperature_source
        if source not in state:
            raise KeyError(
                f"ThermalStress requires the declared input '{source}' [K]; "
                f"available keys: {sorted(state)}"
            )
        temperature = state[source]
        
        # Ensure temperature is a tensor on correct device
        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor(temperature, device=self.device)
        else:
            temperature = temperature.to(self.device)
        
        # Get stress factor bounds from parameters
        stress_min = self.get_parameter_value(
            "global.factor_min",
            namespace="biosphere.stress",
        )
        stress_max = self.get_parameter_value(
            "global.factor_max",
            namespace="biosphere.stress",
        )
        
        # Get all temperature parameters dynamically from ParameterManager
        temp_optimum = self.get_parameter_value(
            "optimal_temperature",
            namespace=self.namespace,
        )
        temp_tolerance_width = self.get_parameter_value(
            "temperature_tolerance_width",
            namespace=self.namespace,
        )
        heat_threshold = self.get_parameter_value(
            "heat_stress_threshold",
            namespace=self.namespace,
        )
        heat_coeff = self.get_parameter_value(
            "heat_stress_coefficient",
            namespace=self.namespace,
        )
        critical_high = self.get_parameter_value(
            "critical_high_temperature",
            namespace=self.namespace,
        )
        cold_threshold = self.get_parameter_value(
            "cold_stress_threshold",
            namespace=self.namespace,
        )
        cold_coeff = self.get_parameter_value(
            "cold_stress_coefficient",
            namespace=self.namespace,
        )
        freezing_threshold = self.get_parameter_value(
            "freezing_damage_threshold",
            namespace=self.namespace,
        )
        
        # References: Berry & Bjorkman (1980); Expression: f_temp = exp(-0.5 * ((T - T_opt)/sigma)^2) where sigma = temperature_tolerance_width.
        temp_deviation = (temperature - temp_optimum) / temp_tolerance_width
        thermal_factor = torch.exp(-0.5 * temp_deviation**2)
        
        # References: Salvucci & Crafts-Brandner (2004); Expression: f_heat = max(0, 1 - k_heat * (T - T_threshold)).
        heat_stress = torch.where(
            temperature > heat_threshold,
            torch.maximum(
                torch.zeros_like(temperature),
                1.0 + heat_coeff * (temperature - heat_threshold)
            ),
            torch.ones_like(temperature)
        )
        
        # Apply critical temperature cutoff for heat stress
        heat_stress = torch.where(
            temperature > critical_high,
            torch.zeros_like(temperature),
            heat_stress
        )
        
        # References: Hatfield & Prueger (2015); Expression: f_cold = max(0, 1 - k_cold * (T_cold - T)).
        cold_stress = torch.where(
            temperature < cold_threshold,
            torch.maximum(
                torch.zeros_like(temperature),
                1.0 + cold_coeff * (cold_threshold - temperature)
            ),
            torch.ones_like(temperature)
        )
        
        # Apply freezing damage cutoff for cold stress
        cold_stress = torch.where(
            temperature < freezing_threshold,
            torch.zeros_like(temperature),
            cold_stress
        )
        
        # Clamp all stress factors to valid range
        thermal_factor = torch.clamp(thermal_factor, min=stress_min, max=stress_max)
        heat_stress = torch.clamp(heat_stress, min=stress_min, max=stress_max)
        cold_stress = torch.clamp(cold_stress, min=stress_min, max=stress_max)
        
        # References: SCM variable contracts of temperature_stress_factor,
        #   heat_stress_factor and cold_stress_factor (kind diagnostic,
        #   incrementable false); architecture_overview.md Section 4.2 (update modes).
        # Mathematical expression: y_k = f(T_k) for every governed step k (REPLACE),
        #   with f one of f_T, f_hot, f_cold above; never y_k = y_(k-1) + f(T_k) * dt.
        # Parameter meanings: y_k is the committed factor after step k; T_k is the
        #   air temperature read in that step; dt is the step length.
        # Units: y_k and f [1]; T_k [K]; dt [s].
        # Domain/constraints: per_timestep is False, so the value is not scaled by dt.
        #   A bare tensor would be coerced to an ACCUMULATE update, which the three
        #   contracts refuse; the tensors themselves are passed unchanged.
        # Sign convention: not applicable; the factors are controls, not fluxes.
        return {
            name: UpdateItem(
                value=value,
                mode=UpdateMode.REPLACE,
                per_timestep=False,
                unit='1',
                source_functor=self.name,
            )
            for name, value in (
                ('temperature_stress_factor', thermal_factor),
                ('heat_stress_factor', heat_stress),
                ('cold_stress_factor', cold_stress),
            )
            if name in self.stress_outputs
        }


__all__ = ['ThermalStress']
