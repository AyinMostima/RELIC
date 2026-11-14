"""
Data Assimilation Base Classes and Interfaces

This module provides the fundamental abstractions for data assimilation in the
RELIC optimization framework. It defines the interface contracts and common
functionality shared across different assimilation methods.

Scientific Background:
    Data assimilation combines model forecasts with observations to produce
    optimal estimates of system state. The Bayesian framework provides the
    theoretical foundation where the analysis is the posterior probability
    density function given prior forecast and observation likelihood.

References:
    - Kalnay, E. (2003). Atmospheric Modeling, Data Assimilation and Predictability
    - Evensen, G. (2003). The Ensemble Kalman Filter: theoretical formulation and practical implementation
    - Ide, K. et al. (1997). Unified notation for data assimilation: Operational, sequential and variational
    - Talagrand, O. (1997). Assimilation of observations, an introduction

Mathematical Expression:
    Bayesian Framework: p(x|y) ? p(y|x) * p(x)
    where:
    - p(x|y): posterior (analysis) probability
    - p(y|x): likelihood (observation model)  
    - p(x): prior (forecast) probability
    
    Optimal Analysis: x_a = x_f + K(y - H(x_f))
    where K is the optimal gain matrix
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import torch
from torch import Tensor
import logging

from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager
from relic.parameters import UnifiedParameterManager, get_global_manager
from relic.data.standards.cf_conventions import CFConventions

logger = logging.getLogger(__name__)


class AssimilationMethod(Enum):
    """Available data assimilation methods"""
    ENSEMBLE_KALMAN_FILTER = "enkf"
    EXTENDED_KALMAN_FILTER = "ekf"
    VARIATIONAL_3D = "3dvar"
    VARIATIONAL_4D = "4dvar"
    PARTICLE_FILTER = "pf"


@dataclass
class AssimilationConfig:
    """Configuration for data assimilation system.

    Only settings that an assimilator reads are offered; a quality-control
    threshold, localization radius or output switch that no code applies
    is not accepted (the former fields raised no error and did nothing).
    The method is chosen by the caller (a keyword-only required field; a
    subclass for one method fixes it). ensemble_size None reads
    optimization.assimilation.ensemble_size from the parameter database.
    """
    # Method selection (no default: the caller or the subclass chooses it)
    method: AssimilationMethod = field(kw_only=True)

    # Ensemble settings (for ensemble methods); None reads the YAML value
    ensemble_size: Optional[int] = None

    # Computational settings
    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.ensemble_size is None:
            self.ensemble_size = int(get_global_manager().get(
                "optimization.assimilation.ensemble_size", required=True
            ))


@dataclass
class Observation:
    """
    Single observation data structure.
    
    Encapsulates all information needed to assimilate one observation,
    including value, error statistics, location, time, and metadata.
    
    Attributes:
        value: Observed value (scalar or tensor)
        error_variance: Observation error variance
        coordinates: Spatial coordinates (lat, lon, z)
        time: Observation time 
        variable_name: Name of observed state variable
        quality_flag: Quality control flag (0 = good, 1 = suspect, 2 = bad),
            declared by the caller; it has no default
        metadata: Additional metadata

    The former units field was stored but never compared with the units of
    the observed variable by any observation operator, so it is not offered.
    """
    value: Union[float, Tensor]
    error_variance: Union[float, Tensor]
    coordinates: Tuple[float, float, float]  # (lat, lon, z)
    time: float  # Time in simulation units
    variable_name: str
    quality_flag: int = field(kw_only=True)  # 0=good, 1=suspect, 2=bad
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ObservationSet:
    """Collection of observations for a single assimilation cycle"""
    observations: List[Observation]
    time: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.observations)
    
    def filter_by_quality(self, max_quality: int = 1) -> 'ObservationSet':
        """Filter observations by quality flag"""
        filtered_obs = [obs for obs in self.observations if obs.quality_flag <= max_quality]
        return ObservationSet(observations=filtered_obs, time=self.time, metadata=self.metadata)
    
    def get_observation_vector(self, device: torch.device = None) -> Tensor:
        """Convert observations to tensor vector"""
        values = [obs.value if isinstance(obs.value, (int, float)) 
                 else obs.value.item() for obs in self.observations]
        return torch.tensor(values, device=device)
    
    def get_error_covariance_matrix(self, device: torch.device = None) -> Tensor:
        """Get observation error covariance matrix (diagonal)"""
        variances = [obs.error_variance if isinstance(obs.error_variance, (int, float))
                    else obs.error_variance.item() for obs in self.observations]
        return torch.diag(torch.tensor(variances, device=device))


@dataclass
class AssimilationResult:
    """
    Results from a single assimilation cycle.
    
    Contains the analysis state, diagnostics, and metadata from
    the assimilation process.
    """
    # Analysis results
    analysis_mean: Dict[str, Tensor]  # Analyzed state mean
    analysis_ensemble: Optional[Dict[str, Tensor]] = None  # Full ensemble if saved
    
    # Prior state (for comparison)
    forecast_mean: Dict[str, Tensor] = field(default_factory=dict)
    forecast_ensemble: Optional[Dict[str, Tensor]] = None
    
    # Assimilation diagnostics
    innovation: Optional[Tensor] = None  # y - H(x_f)
    innovation_covariance: Optional[Tensor] = None  # HPH^T + R
    kalman_gain: Optional[Tensor] = None  # K matrix
    
    # Quality metrics and cycle metadata: every producer states them (no
    # zero is reported for a value that was not computed). rms_innovation is
    # NaN when no observation was assimilated.
    rms_innovation: float = field(kw_only=True)
    assimilation_time: float = field(kw_only=True)
    computational_cost: float = field(kw_only=True)
    observations_assimilated: int = field(kw_only=True)
    observations_rejected: int = field(kw_only=True)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ObservationOperator(ABC):
    """
    Abstract base class for observation operators.
    
    The observation operator H maps from model state space to observation
    space, enabling comparison between model predictions and observations.
    """
    
    @abstractmethod
    def forward(self, state: Dict[str, Tensor]) -> Tensor:
        """
        Map model state to observation space.
        
        Parameters:
            state: Model state variables
            
        Returns:
            Predicted observations
        """
    
    @abstractmethod 
    def get_jacobian(self, state: Dict[str, Tensor]) -> Tensor:
        """
        Compute Jacobian matrix partialH/partialx.
        
        Parameters:
            state: Model state variables
            
        Returns:
            Jacobian matrix
        """
    
    def tangent_linear(self, state: Dict[str, Tensor], perturbation: Dict[str, Tensor]) -> Tensor:
        """
        Apply tangent linear model H(x + deltax) ~= H(x) + H'(x)deltax
        
        Default implementation uses Jacobian multiplication.
        Can be overridden for efficiency.
        """
        jacobian = self.get_jacobian(state)
        pert_vector = torch.cat([perturbation[var].flatten() for var in sorted(perturbation.keys())])
        return torch.matmul(jacobian, pert_vector)


class IdentityOperator(ObservationOperator):
    """
    Identity observation operator for direct state observation.
    
    Used when observations directly correspond to state variables
    without spatial/temporal transformation.
    """
    
    def __init__(self, variable_names: List[str]):
        self.variable_names = variable_names
        
    def forward(self, state: Dict[str, Tensor]) -> Tensor:
        """Direct extraction of observed variables"""
        observations = []
        for var_name in self.variable_names:
            # An observed variable missing from the state is an error; it is
            # never observed as zero.
            if var_name not in state:
                raise KeyError(
                    f"Observed variable '{var_name}' not found in state"
                )
            observations.append(state[var_name].flatten())
        
        return torch.cat(observations)
    
    def get_jacobian(self, state: Dict[str, Tensor]) -> Tensor:
        """Identity Jacobian for direct observations"""
        total_size = sum(state[var].numel() for var in self.variable_names if var in state)
        obs_size = len(self.variable_names)
        
        jacobian = torch.zeros(obs_size, total_size)
        
        # Fill diagonal for observed variables
        obs_idx = 0
        state_idx = 0
        for var_name in sorted(state.keys()):
            var_size = state[var_name].numel()
            if var_name in self.variable_names:
                jacobian[obs_idx, state_idx] = 1.0
                obs_idx += 1
            state_idx += var_size
            
        return jacobian


class Assimilator(ABC):
    """
    Abstract base class for data assimilation algorithms.
    
    Defines the common interface for all assimilation methods including
    ensemble and variational approaches.
    """
    
    def __init__(
        self,
        simulator: RELICSimulator,
        state_manager: StateManager,
        parameter_manager: UnifiedParameterManager,
        config: AssimilationConfig
    ):
        """
        Initialize base assimilator.
        
        Parameters:
            simulator: Model simulator
            state_manager: State management system
            parameter_manager: Parameter management system
            config: Assimilation configuration
        """
        self.simulator = simulator
        self.state_manager = state_manager
        self.parameter_manager = parameter_manager
        self.config = config
        
        # Setup device and precision
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        
        # CF conventions for validation
        self.cf_conventions = CFConventions()
        
        # Runtime statistics
        self.cycle_count = 0
        self.total_observations = 0
        self.total_computational_cost = 0.0
        
        logger.info(f"Assimilator initialized with method={config.method.value}")
    
    @abstractmethod
    def assimilate(
        self,
        forecast_state: Dict[str, Tensor],
        observations: ObservationSet,
        observation_operator: ObservationOperator,
        background_covariance: Optional[Tensor] = None
    ) -> AssimilationResult:
        """
        Perform data assimilation for one cycle.
        
        Parameters:
            forecast_state: Prior state from model forecast
            observations: Observations to assimilate
            observation_operator: Maps state to observation space
            background_covariance: Background error covariance (if available)
            
        Returns:
            Assimilation result with analysis state and diagnostics
        """
    
    def validate_observations(self, observations: ObservationSet) -> ObservationSet:
        """
        Apply quality control to observations.
        
        Parameters:
            observations: Raw observations
            
        Returns:
            Quality-controlled observations
        """
        valid_observations = []
        rejected_count = 0
        
        for obs in observations.observations:
            # Check for reasonable values
            if isinstance(obs.value, (int, float)):
                if np.isnan(obs.value) or np.isinf(obs.value):
                    rejected_count += 1
                    continue
            else:
                if torch.isnan(obs.value).any() or torch.isinf(obs.value).any():
                    rejected_count += 1
                    continue
                    
            # Check observation error
            if isinstance(obs.error_variance, (int, float)):
                if obs.error_variance <= 0 or np.isnan(obs.error_variance):
                    rejected_count += 1
                    continue
            else:
                if (obs.error_variance <= 0).any() or torch.isnan(obs.error_variance).any():
                    rejected_count += 1
                    continue
                    
            # Apply quality flag filter
            if obs.quality_flag > 1:  # Reject bad observations
                rejected_count += 1
                continue
                
            valid_observations.append(obs)
        
        if rejected_count > 0:
            logger.info(f"Quality control rejected {rejected_count}/{len(observations)} observations")
        
        return ObservationSet(
            observations=valid_observations,
            time=observations.time,
            metadata=observations.metadata
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get assimilation system statistics"""
        avg_cost = self.total_computational_cost / max(self.cycle_count, 1)
        return {
            "cycles_completed": self.cycle_count,
            "total_observations": self.total_observations,
            "average_computational_cost": avg_cost,
            "total_computational_cost": self.total_computational_cost
        }
