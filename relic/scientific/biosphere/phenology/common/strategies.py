"""
Phenological Strategy Patterns

This module implements the Strategy pattern for phenological processes,
providing reusable, composable strategies for growth factors, senescence
rates, and phenological transitions. This reduces code duplication and
enables flexible model composition.

Strategy Types:
    - GrowthFactorStrategy: Controls growth activity based on environmental cues
    - SenescenceStrategy: Controls leaf senescence and nutrient reallocation  
    - TransitionStrategy: Manages smooth phenological phase transitions
    - PhenologyStrategy: Composite strategy combining multiple components

All strategies support PyTorch tensors, automatic differentiation, and
batch processing for integration with the RELIC framework.

Design Pattern:
    Each strategy implements a common interface with compute() method that
    takes environmental inputs and returns phenological factors (0-1 scale).
    Strategies can be combined, chained, and parameterized independently.

References:
    - Gamma, E., et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software
    - Martin, R.C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any
from enum import Enum
import torch
from torch import Tensor

from .response_curves import (
    sigmoid_response, tanh_response, double_sigmoid_response,
    linear_threshold_response
)
from relic.parameters import UnifiedParameterManager
from .....core.abstractions import StateDict
from .....utils.physical_constants import get_physical_constant

T_FREEZE = get_physical_constant("T_FREEZE")

TensorLike = Union[Tensor, float, int]


class StrategyType(Enum):
    """Enumeration of strategy types for phenological processes."""
    GROWTH_FACTOR = "growth_factor"
    SENESCENCE = "senescence"
    TRANSITION = "transition"
    COMPOSITE = "composite"


class PhenologyStrategy(ABC):
    """
    Abstract base class for phenological strategies.
    
    Defines the common interface that all phenological strategies must
    implement. Strategies compute phenological factors (0-1) based on
    environmental inputs and internal parameters.
    
    Mathematical Expression:
        f: Environment * Parameters ? [0,1]
        
        Where f is the strategy function that maps environmental conditions
        and strategy parameters to a phenological factor between 0 and 1.
    """
    
    def __init__(
        self,
        name: str,
        strategy_type: StrategyType,
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        if parameter_manager is None:
            raise TypeError("parameter_manager is required")
        self.name = name
        self.strategy_type = strategy_type
        self.parameter_manager = parameter_manager
        self.parameters = {}
        # Set parameter namespace for unified parameter access
        self.parameter_namespace = "biosphere.phenology"

    def get_parameter_value(
        self,
        name: str,
        namespace: Optional[str] = None,
        required: bool = True,
        units: Optional[str] = None,
        expected_type: Optional[type] = None,
    ) -> Any:
        """Read a formal YAML key through the explicitly injected manager.

        The key is required: a missing key raises. A caller reads an optional parameter
        with required=False and handles its absence (None) explicitly; no caller-supplied
        default value exists.
        """
        return self.parameter_manager.get(
            name,
            namespace=namespace or self.parameter_namespace,
            units=units,
            expected_type=expected_type,
            required=required,
        )
    
    @abstractmethod
    def compute(self, inputs: StateDict, **kwargs) -> Tensor:
        """
        Compute phenological factor based on environmental conditions.
        
        Parameters:
            inputs : StateDict
                Declared phenology inputs (a subset of the state mapping, for
                example day_of_year, air_temperature, accumulated_gdd,
                soil_water_potential); no separate input channel
            **kwargs : dict
                Additional strategy-specific parameters
                
        Returns:
            Tensor: Phenological factor(s) in range [0,1]
        """
    
    def set_parameter(self, name: str, value: Any) -> None:
        """Set a strategy parameter."""
        self.parameters[name] = value
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a strategy parameter."""
        return self.parameters.get(name, default)


class GrowthFactorStrategy(PhenologyStrategy):
    """
    Base class for growth factor strategies.
    
    Growth factors control the activity level of vegetative growth processes
    including leaf expansion, biomass allocation, and photosynthetic capacity.
    Values range from 0 (dormant) to 1 (maximum growth activity).
    """
    
    def __init__(
        self,
        name: str = "growth_factor",
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        super().__init__(
            name,
            StrategyType.GROWTH_FACTOR,
            parameter_manager=parameter_manager,
        )


class StaticGrowthStrategy(GrowthFactorStrategy):
    """
    Static calendar date-based growth factor strategy.
    
    Uses fixed calendar dates with smooth sigmoid transitions for
    growth onset and cessation. Simple but effective for regions
    with predictable seasonal patterns.
    
    Parameters:
        onset_doy : int
            Day of year for growth onset
        offset_doy : int  
            Day of year for growth cessation
        steepness : float
            Transition steepness parameter
    """
    
    def __init__(
        self,
        onset_doy: Optional[int] = None,
        offset_doy: Optional[int] = None,
        steepness: Optional[float] = None,
        name: str = "static_growth",
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        super().__init__(name, parameter_manager=parameter_manager)
        # Store any provided overrides, but prefer ParameterManager
        if onset_doy is not None:
            self.set_parameter('onset_doy_override', onset_doy)
        if offset_doy is not None:
            self.set_parameter('offset_doy_override', offset_doy)
        if steepness is not None:
            self.set_parameter('steepness_override', steepness)
    
    def compute(self, inputs: StateDict, **kwargs) -> Tensor:
        """Compute static growth factor based on day of year."""
        # Day of year is a required time-varying variable
        if 'day_of_year' not in inputs:
            raise ValueError("day_of_year is required in inputs but not provided")
        day_of_year = inputs['day_of_year']
        
        # Get parameters from ParameterManager or use overrides
        onset_doy = self.get_parameter('onset_doy_override', None)
        if onset_doy is None:
            onset_doy = self.get_parameter_value('leaf_onset_doy', namespace="biosphere.phenology.static")
        
        offset_doy = self.get_parameter('offset_doy_override', None)
        if offset_doy is None:
            offset_doy = self.get_parameter_value('leaf_offset_doy', namespace="biosphere.phenology.static")
        
        steepness = self.get_parameter('steepness_override', None)
        if steepness is None:
            steepness = self.get_parameter_value('transition_steepness', namespace="biosphere.phenology.static")
        
        # Double sigmoid for seasonal growth pattern
        return double_sigmoid_response(
            day_of_year,
            x_onset=onset_doy,
            x_offset=offset_doy,
            steepness_onset=steepness,
            steepness_offset=steepness,
            y_base=0.0,
            y_peak=1.0
        )


class GDDGrowthStrategy(GrowthFactorStrategy):
    """
    Growing Degree Day (GDD) based growth factor strategy.
    
    Growth activity is controlled by accumulated thermal time above
    a base temperature threshold. Provides temperature sensitivity
    for phenological timing.
    
    Parameters:
        gdd_start : float
            GDD threshold for growth initiation
        gdd_end : float
            GDD threshold for growth cessation
        base_temp : float
            Base temperature for GDD calculation
        steepness : float
            Transition steepness
    """
    
    def __init__(
        self,
        gdd_start: Optional[float] = None,
        gdd_end: Optional[float] = None,
        base_temp: Optional[float] = None,
        steepness: Optional[float] = None,
        name: str = "gdd_growth",
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        super().__init__(name, parameter_manager=parameter_manager)
        # Store any provided overrides, but prefer ParameterManager
        if gdd_start is not None:
            self.set_parameter('gdd_start_override', gdd_start)
        if gdd_end is not None:
            self.set_parameter('gdd_end_override', gdd_end)
        if base_temp is not None:
            self.set_parameter('base_temp_override', base_temp)
        if steepness is not None:
            self.set_parameter('steepness_override', steepness)
    
    def compute(self, inputs: StateDict, **kwargs) -> Tensor:
        """Compute GDD-based growth factor."""
        # Get accumulated GDD from the inputs or compute from temperature
        if 'accumulated_gdd' in inputs:
            gdd_total = inputs['accumulated_gdd']
        else:
            # Fallback: use simple GDD approximation
            # Air temperature is required for GDD calculation
            if 'air_temperature' not in inputs:
                raise ValueError("air_temperature is required for GDD calculation but not provided")
            temp = inputs['air_temperature']  # K
            # Get base temp from ParameterManager or use override
            base_temp_celsius = self.get_parameter('base_temp_override', None)
            if base_temp_celsius is None:
                base_temp_celsius = self.get_parameter_value('base_temp', namespace="biosphere.phenology.gdd")
            base_temp_kelvin = base_temp_celsius + T_FREEZE  # Convert to K
            # Get clamp minimum from parameter manager
            clamp_min = self.get_parameter_value('clamp_min', namespace="biosphere.phenology.computational")
            daily_gdd = torch.clamp(temp - base_temp_kelvin, min=clamp_min)
            gdd_total = daily_gdd  # Single day approximation
        
        # Get parameters from ParameterManager or use overrides
        gdd_start = self.get_parameter('gdd_start_override', None)
        if gdd_start is None:
            gdd_start = self.get_parameter_value('threshold_start', namespace="biosphere.phenology.gdd")
        
        gdd_end = self.get_parameter('gdd_end_override', None)
        if gdd_end is None:
            gdd_end = self.get_parameter_value('threshold_end', namespace="biosphere.phenology.gdd")
        
        steepness = self.get_parameter('steepness_override', None) 
        if steepness is None:
            steepness = self.get_parameter_value('transition_steepness', namespace="biosphere.phenology.static")
        
        # Double sigmoid response for GDD-based growth
        return double_sigmoid_response(
            gdd_total,
            x_onset=gdd_start,
            x_offset=gdd_end,
            steepness_onset=steepness,
            steepness_offset=steepness,
            y_base=0.0,
            y_peak=1.0
        )


class SenescenceStrategy(PhenologyStrategy):
    """
    Base class for senescence strategies.
    
    Senescence strategies control leaf aging, nutrient reallocation,
    and biomass turnover rates. Higher values indicate more active
    senescence processes.
    """
    
    def __init__(
        self,
        name: str = "senescence",
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        super().__init__(
            name,
            StrategyType.SENESCENCE,
            parameter_manager=parameter_manager,
        )


class TimeBasedSenescenceStrategy(SenescenceStrategy):
    """
    Time-based senescence strategy using day of year.
    
    Simple senescence model based on calendar dates with smooth
    transitions. Senescence rate increases as day of year approaches
    and passes the senescence start date.
    
    Parameters:
        senescence_start_doy : int
            Day of year when senescence begins
        senescence_end_doy : int
            Day of year when senescence peaks
        steepness : float
            Rate of senescence increase
    """
    
    def __init__(
        self,
        senescence_start_doy: Optional[int] = None,
        senescence_end_doy: Optional[int] = None,
        steepness: Optional[float] = None,
        name: str = "time_senescence",
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        super().__init__(name, parameter_manager=parameter_manager)
        # Store any provided overrides
        if senescence_start_doy is not None:
            self.set_parameter('senescence_start_doy_override', senescence_start_doy)
        if senescence_end_doy is not None:
            self.set_parameter('senescence_end_doy_override', senescence_end_doy)
        if steepness is not None:
            self.set_parameter('steepness_override', steepness)
    
    def compute(self, inputs: StateDict, **kwargs) -> Tensor:
        """Compute time-based senescence rate."""
        # Day of year is a required time-varying variable
        if 'day_of_year' not in inputs:
            raise ValueError("day_of_year is required in inputs but not provided")
        day_of_year = inputs['day_of_year']
        
        # Get parameters from ParameterManager or use overrides
        start_doy = self.get_parameter('senescence_start_doy_override', None)
        if start_doy is None:
            # Using a reasonable default for senescence start (day 240 = late August)
            start_doy = self.get_parameter_value('senescence_start_doy', namespace="biosphere.phenology.turnover")
        
        end_doy = self.get_parameter('senescence_end_doy_override', None)
        if end_doy is None:
            # Using a reasonable default for senescence end (day 310 = early November)
            end_doy = self.get_parameter_value('senescence_end_doy', namespace="biosphere.phenology.turnover")
        
        steepness = self.get_parameter('steepness_override', None)
        if steepness is None:
            steepness = self.get_parameter_value('transition_steepness', namespace="biosphere.phenology.static")
        
        # Linear ramp from start to end, then constant
        return linear_threshold_response(
            day_of_year,
            x_min=start_doy,
            x_max=end_doy,
            y_min=0.0,
            y_max=1.0
        )


class DroughtSenescenceStrategy(SenescenceStrategy):
    """
    Drought-induced senescence strategy.
    
    Senescence triggered by water stress, independent of photoperiod.
    Used for drought-deciduous vegetation and stress-response modeling.
    
    Parameters:
        swp_threshold : float
            Soil water potential threshold for senescence [MPa]
        swp_severe : float  
            Severe drought threshold for maximum senescence [MPa]
        recovery_rate : float
            Rate of senescence recovery when water stress relieved
    """
    
    def __init__(
        self,
        swp_threshold: Optional[float] = None,
        swp_severe: Optional[float] = None,
        recovery_rate: Optional[float] = None,
        name: str = "drought_senescence",
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        super().__init__(name, parameter_manager=parameter_manager)
        # Store any provided overrides
        if swp_threshold is not None:
            self.set_parameter('swp_threshold_override', swp_threshold)
        if swp_severe is not None:
            self.set_parameter('swp_severe_override', swp_severe)
        if recovery_rate is not None:
            self.set_parameter('recovery_rate_override', recovery_rate)
    
    def compute(self, inputs: StateDict, **kwargs) -> Tensor:
        """Compute drought-induced senescence rate."""
        # Soil water potential is required for calculation
        if 'soil_water_potential' not in inputs:
            raise ValueError("soil_water_potential is required but not provided")
        swp = inputs['soil_water_potential']
        
        # Get parameters from ParameterManager or use overrides
        threshold = self.get_parameter('swp_threshold_override', None)
        if threshold is None:
            threshold = self.get_parameter_value('drought_stress_threshold', namespace="biosphere.phenology.stress")
        
        severe = self.get_parameter('swp_severe_override', None)
        if severe is None:
            # Use a more severe threshold for complete senescence
            severe = self.get_parameter_value('swp_severe', namespace="biosphere.phenology.stress")
        
        # Linear response between threshold and severe drought
        return linear_threshold_response(
            swp,
            x_min=severe,    # More negative = more severe
            x_max=threshold, # Less negative = less stress
            y_min=1.0,       # Maximum senescence at severe drought
            y_max=0.0        # No senescence above threshold
        )


class TransitionStrategy(PhenologyStrategy):
    """
    Base class for phenological transition strategies.
    
    Transition strategies manage smooth changes between phenological
    phases to avoid discontinuities in carbon allocation and other
    ecosystem processes.
    """
    
    def __init__(
        self,
        name: str = "transition",
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        super().__init__(
            name,
            StrategyType.TRANSITION,
            parameter_manager=parameter_manager,
        )


class SmoothTransitionStrategy(TransitionStrategy):
    """
    Smooth transition strategy using sigmoid functions.
    
    Provides smooth transitions between phenological states using
    configurable sigmoid or tanh functions. Prevents numerical
    discontinuities and unrealistic rapid changes.
    
    Parameters:
        transition_type : str
            Type of transition: 'sigmoid', 'tanh', 'linear'
        steepness : float
            Transition steepness parameter
        center_point : float
            Center point of transition
    """
    
    def __init__(
        self,
        transition_type: Optional[str] = None,
        steepness: Optional[float] = None,
        center_point: Optional[float] = None,
        name: str = "smooth_transition",
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        super().__init__(name, parameter_manager=parameter_manager)
        # Store any provided overrides
        if transition_type is not None:
            self.set_parameter('transition_type_override', transition_type)
        if steepness is not None:
            self.set_parameter('steepness_override', steepness)
        if center_point is not None:
            self.set_parameter('center_point_override', center_point)
    
    def compute(self, inputs: StateDict, **kwargs) -> Tensor:
        """Compute smooth transition factor."""
        # Use day of year as default transition variable
        # Day of year is the default transition variable
        if 'day_of_year' not in inputs:
            raise ValueError("day_of_year is required as transition variable but not provided")
        x = inputs['day_of_year']
        
        # Allow override of transition variable
        if 'transition_variable' in kwargs:
            x = kwargs['transition_variable']
        
        # Get parameters from overrides or defaults
        transition_type = self.get_parameter('transition_type_override', None)
        if transition_type is None:
            transition_type = 'sigmoid'  # Default transition type
        
        steepness = self.get_parameter('steepness_override', None)
        if steepness is None:
            steepness = self.get_parameter_value('sigmoid_steepness', namespace="biosphere.phenology.static")
        
        center = self.get_parameter('center_point_override', None)
        if center is None:
            center = self.get_parameter_value('sigmoid_midpoint', namespace="biosphere.phenology.static")
        
        if transition_type == 'sigmoid':
            return sigmoid_response(x, center, steepness, 0.0, 1.0)
        elif transition_type == 'tanh':
            return tanh_response(x, center, steepness, 0.0, 1.0)
        elif transition_type == 'linear':
            # Simple linear transition over +/-1/steepness range
            half_width = 1.0 / max(steepness, 1e-6)
            return linear_threshold_response(
                x, center - half_width, center + half_width, 0.0, 1.0
            )
        else:
            raise ValueError(f"Unknown transition type: {transition_type}")


class CompositeStrategy(PhenologyStrategy):
    """
    Composite strategy combining multiple component strategies.
    
    Enables complex phenological responses by combining simpler
    strategies using mathematical operations (multiply, add, min, max).
    Follows the Composite design pattern.
    
    Operations:
        - multiply: Product of component strategies (intersection)
        - add: Sum of strategies (union, clamped to [0,1])
        - min: Minimum of strategies (most limiting)
        - max: Maximum of strategies (most permissive)
        - weighted_sum: Weighted linear combination
    
    Example:
        >>> # Combine temperature and moisture limitations
        >>> temp_strategy = GDDGrowthStrategy(parameter_manager=manager)
        >>> drought_strategy = DroughtSenescenceStrategy(parameter_manager=manager)
        >>> composite = CompositeStrategy(
        ...     strategies=[temp_strategy, drought_strategy],
        ...     operation='multiply',
        ...     weights=[1.0, -1.0],  # Growth * (1 - senescence)
        ...     parameter_manager=manager,
        ... )
    """
    
    def __init__(
        self,
        strategies: List[PhenologyStrategy],
        operation: str = "multiply",
        weights: Optional[List[float]] = None,
        name: str = "composite",
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        super().__init__(
            name,
            StrategyType.COMPOSITE,
            parameter_manager=parameter_manager,
        )
        
        if not strategies:
            raise ValueError("At least one strategy must be provided")
        
        valid_operations = ['multiply', 'add', 'min', 'max', 'weighted_sum']
        if operation not in valid_operations:
            raise ValueError(f"Operation must be one of: {valid_operations}")
        
        self.strategies = strategies
        self.set_parameter('operation', operation)
        
        if weights is not None:
            if len(weights) != len(strategies):
                raise ValueError("Number of weights must match number of strategies")
            self.set_parameter('weights', weights)
        else:
            self.set_parameter('weights', [1.0] * len(strategies))
    
    def compute(self, inputs: StateDict, **kwargs) -> Tensor:
        """Compute composite strategy response."""
        # Compute all component strategies
        results = []
        for strategy in self.strategies:
            result = strategy.compute(inputs, **kwargs)
            results.append(result)
        
        if not results:
            zero_value = self.get_parameter_value("zero_initial", namespace="biosphere.phenology.computational")
            return zero_value
        
        operation = self.get_parameter('operation', 'multiply')
        weights = self.get_parameter('weights', [1.0] * len(results))
        
        if operation == 'multiply':
            # Product of all strategies
            output = results[0]
            for result in results[1:]:
                output = output * result
            
        elif operation == 'add':
            # Sum with clamping to [0,1]
            output = torch.sum(torch.stack(results), dim=0)
            clamp_min = self.get_parameter_value('clamp_min', namespace="biosphere.phenology.computational")
            clamp_max = self.get_parameter_value('clamp_max', namespace="biosphere.phenology.computational")
            output = torch.clamp(output, min=clamp_min, max=clamp_max)
            
        elif operation == 'min':
            # Most limiting factor
            output = torch.min(torch.stack(results), dim=0)[0]
            
        elif operation == 'max':
            # Most permissive factor
            output = torch.max(torch.stack(results), dim=0)[0]
            
        elif operation == 'weighted_sum':
            # Weighted linear combination
            weighted_results = []
            for result, weight in zip(results, weights):
                weighted_results.append(weight * result)
            output = torch.sum(torch.stack(weighted_results), dim=0)
            clamp_min = self.get_parameter_value('clamp_min', namespace="biosphere.phenology.computational")
            clamp_max = self.get_parameter_value('clamp_max', namespace="biosphere.phenology.computational")
            output = torch.clamp(output, min=clamp_min, max=clamp_max)
            
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        return output
    
    def add_strategy(self, strategy: PhenologyStrategy, weight: float = 1.0) -> None:
        """Add a new component strategy."""
        self.strategies.append(strategy)
        weights = self.get_parameter('weights', [])
        weights.append(weight)
        self.set_parameter('weights', weights)
    
    def remove_strategy(self, index: int) -> None:
        """Remove a component strategy by index."""
        if 0 <= index < len(self.strategies):
            del self.strategies[index]
            weights = self.get_parameter('weights', [])
            if index < len(weights):
                del weights[index]
                self.set_parameter('weights', weights)


# Factory functions for common strategy configurations
def create_temperate_deciduous_strategy(
    *,
    parameter_manager: UnifiedParameterManager,
) -> CompositeStrategy:
    """Create strategy for temperate deciduous trees."""
    growth_strategy = GDDGrowthStrategy(
        gdd_start=150.0,
        gdd_end=1000.0,
        base_temp=5.0,
        parameter_manager=parameter_manager,
    )
    
    senescence_strategy = TimeBasedSenescenceStrategy(
        senescence_start_doy=240,
        senescence_end_doy=300,
        parameter_manager=parameter_manager,
    )
    
    return CompositeStrategy(
        strategies=[growth_strategy, senescence_strategy],
        operation='multiply',
        weights=[1.0, -1.0],  # Growth * (1 - senescence)
        name='temperate_deciduous',
        parameter_manager=parameter_manager,
    )


def create_crop_strategy(
    crop_type: str = "wheat",
    *,
    parameter_manager: UnifiedParameterManager,
) -> CompositeStrategy:
    """Create strategy for agricultural crops."""
    if crop_type == "wheat":
        growth_strategy = GDDGrowthStrategy(
            gdd_start=100.0,
            gdd_end=1200.0,
            base_temp=0.0,
            parameter_manager=parameter_manager,
        )
    else:
        # Default crop parameters
        growth_strategy = GDDGrowthStrategy(
            parameter_manager=parameter_manager
        )
    
    # Simple time-based senescence for crops
    senescence_strategy = TimeBasedSenescenceStrategy(
        senescence_start_doy=220,
        senescence_end_doy=270,
        parameter_manager=parameter_manager,
    )
    
    return CompositeStrategy(
        strategies=[growth_strategy, senescence_strategy],
        operation='multiply',
        weights=[1.0, -1.0],
        name=f'{crop_type}_crop',
        parameter_manager=parameter_manager,
    )
