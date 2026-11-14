"""
Unified Abstract Interfaces for Stochastic Processes

This module defines category-theory inspired abstract base classes for
all stochastic processes in the RELIC framework. It provides unified
interfaces for parameter sampling, ensemble execution, and validation
metrics.

References:
    - Awodey, S. (2010). Category Theory. Oxford University Press.
    - Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules,
      prediction, and estimation. J. Am. Stat. Assoc. 102, 359-378.
    - Gardiner, C. (2009). Stochastic Methods: A Handbook for the Natural
      and Social Sciences. Springer, 4th edition.

Mathematical Framework:
    Objects: State spaces (parameter spaces, observational spaces)
    Morphisms: Stochastic processes as arrows between objects
    Composition: Sequential application of stochastic transformations
    
Authors: Yuchong Long
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
from dataclasses import dataclass


@dataclass
class ProcessMetadata:
    """
    Metadata container for stochastic processes.
    
    Parameters:
        name (str): Process identifier
        description (str): Human-readable description
        units (Dict[str, str]): CF-1.8 compliant units for each variable
        parameters (Dict[str, Any]): Process parameters with bounds/defaults
        references (str): Scientific literature citations
        version (str): Implementation version
    """
    name: str
    description: str
    units: Dict[str, str]
    parameters: Dict[str, Any] 
    references: str
    version: str = "1.0.0"


class StochasticProcess(ABC):
    """
    Abstract base class for all stochastic processes in RELIC.
    
    This class defines the fundamental interface for stochastic transformations
    following category theory principles. Each process is a morphism that maps
    from an input state space to an output state space with uncertainty.
    
    Mathematical Expression:
        F: (X, theta, omega) -> Y
        where:
        - X: input state tensor
        - theta: parameter tensor  
        - omega: random realization
        - Y: output state tensor (stochastic)
    
    Units:
        All tensors must carry CF-1.8 compliant units metadata
        
    References:
        - Gardiner, C. (2009). Stochastic Methods. Springer.
        - Kloeden, P.E. & Platen, E. (1992). Numerical Solution of 
          Stochastic Differential Equations. Springer.
    """
    
    def __init__(self, metadata: ProcessMetadata):
        """
        Initialize stochastic process with metadata.
        
        Parameters:
            metadata (ProcessMetadata): Process description and parameters
        """
        self.metadata = metadata
        self._validate_metadata()
    
    @abstractmethod
    def sample(self, 
               input_state: torch.Tensor,
               num_samples: int = 1,
               seed: Optional[int] = None,
               **kwargs) -> Dict[str, torch.Tensor]:
        """
        Generate stochastic samples from the process.
        
        Parameters:
            input_state (torch.Tensor): Input state variables [batch_size, ...]
            num_samples (int): Number of Monte Carlo samples to generate
            seed (Optional[int]): Random seed for reproducibility
            **kwargs: Process-specific parameters
            
        Returns:
            Dict[str, torch.Tensor]: Sampled outputs with CF-1.8 variable names
                Format: {variable_name: tensor[num_samples, batch_size, ...]}
                
        Mathematical Expression:
            Y_i ~ F(X, theta, omega_i) for i = 1, ..., num_samples
            
        Units:
            Output tensors inherit appropriate CF-1.8 units from metadata
        """
    
    @abstractmethod
    def log_probability(self,
                       input_state: torch.Tensor,
                       output_state: torch.Tensor,
                       **kwargs) -> torch.Tensor:
        """
        Compute log probability density of output given input.
        
        Parameters:
            input_state (torch.Tensor): Input state variables
            output_state (torch.Tensor): Observed output variables  
            **kwargs: Process-specific parameters
            
        Returns:
            torch.Tensor: Log probability density [batch_size]
            
        Mathematical Expression:
            log p(Y|X, theta) = log p(output_state | input_state, parameters)
            
        Units:
            Dimensionless (log probability)
        """
    
    @abstractmethod 
    def moments(self,
                input_state: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute analytical moments (mean, variance) if available.
        
        Parameters:
            input_state (torch.Tensor): Input state variables
            **kwargs: Process-specific parameters
            
        Returns:
            Dict[str, torch.Tensor]: Moments for each output variable
                Format: {variable_name: {mean: tensor, var: tensor}}
                
        Mathematical Expression:
            mu = E[Y|X, theta], sigma^2 = Var[Y|X, theta]
            
        Units:
            Mean: same as output variable
            Variance: (output units)^2
        """
    
    def describe(self) -> Dict[str, Any]:
        """
        Get complete process description including parameters and units.
        
        Returns:
            Dict[str, Any]: Complete process metadata and state
        """
        return {
            "name": self.metadata.name,
            "description": self.metadata.description, 
            "units": self.metadata.units,
            "parameters": self.metadata.parameters,
            "references": self.metadata.references,
            "version": self.metadata.version,
            "cf_compliant": False,
            "cf_validation_scope": "not_evaluated",
            "dataset_conformance_checked": False,
        }
    
    def _validate_metadata(self):
        """Validate required process-metadata fields without CF claims."""
        required_fields = ["name", "description", "units", "parameters", "references"]
        for field in required_fields:
            if not hasattr(self.metadata, field):
                raise ValueError(f"Missing required metadata field: {field}")
                
        # Validate CF-1.8 units
        for var, unit in self.metadata.units.items():
            if not isinstance(unit, str):
                raise ValueError(f"Units must be strings (CF-1.8): {var} = {unit}")


class ValidationMetrics(ABC):
    """
    Abstract base class for validation metrics and scoring rules.
    
    Implements proper scoring rules for probabilistic forecast evaluation
    following Gneiting & Raftery (2007). All metrics are strictly proper
    scoring rules that provide unbiased evaluation.
    
    Mathematical Framework:
        S(F, y) = score for forecast distribution F and observation y
        where S is strictly proper: E[S(F, Y)] <= E[S(G, Y)] iff F = G
        
    References:
        - Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules.
          J. Am. Stat. Assoc. 102, 359-378.
        - Wilks, D.S. (2011). Statistical Methods in the Atmospheric Sciences.
          Academic Press, 3rd edition.
    """
    
    @abstractmethod
    def compute_score(self,
                     observations: torch.Tensor,
                     predictions: Union[torch.Tensor, Dict[str, torch.Tensor]], 
                     **kwargs) -> Dict[str, float]:
        """
        Compute validation score for predictions vs observations.
        
        Parameters:
            observations (torch.Tensor): True values [n_obs, ...]
            predictions: Forecast ensemble or distribution parameters
            **kwargs: Metric-specific options
            
        Returns:
            Dict[str, float]: Computed scores and diagnostics
            
        Mathematical Expression:
            S = (1/n) Sigma? S(F_i, y_i)
            
        Units:
            Dimensionless (standardized scores)
        """
    
    @abstractmethod
    def diagnostic_plots(self,
                        observations: torch.Tensor,
                        predictions: Union[torch.Tensor, Dict[str, torch.Tensor]],
                        save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate diagnostic plots for validation assessment.
        
        Parameters:
            observations (torch.Tensor): True values
            predictions: Forecast ensemble or parameters
            save_path (Optional[str]): Path to save plots
            
        Returns:
            Dict[str, Any]: Plot data and statistics for further analysis
        """


class ParameterUncertainty(StochasticProcess):
    """
    Abstract base class for parameter uncertainty quantification.
    
    Handles sampling from parameter distributions with optional spatial
    correlation structure. Integrates with parameters/manager.py for
    unified parameter handling.
    
    Mathematical Expression:
        theta ~ pi(theta | bounds, correlation)
        where pi is the parameter prior distribution
        
    References:
        - Saltelli, A. et al. (2008). Global Sensitivity Analysis: 
          The Primer. Wiley.
        - Rosolem, R. et al. (2012). A fully multiple-criteria implementation
          of the Sobol method. J. Geophys. Res. 117, D07103.
    """
    
    @abstractmethod
    def sample_parameters(self,
                         param_specs: Dict[str, Any],
                         num_samples: int,
                         spatial_correlation: Optional[torch.Tensor] = None,
                         seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Sample parameters from specified distributions.
        
        Parameters:
            param_specs (Dict[str, Any]): Parameter specifications from manager
            num_samples (int): Number of parameter sets to sample
            spatial_correlation (Optional[torch.Tensor]): Spatial correlation matrix
            seed (Optional[int]): Random seed
            
        Returns:
            Dict[str, torch.Tensor]: Sampled parameter sets
                Format: {param_name: tensor[num_samples, spatial_dims...]}
        """


class WeatherStochasticity(StochasticProcess):
    """
    Abstract base class for stochastic weather generation.
    
    Generates synthetic weather time series that preserve statistical
    properties of observed climate while providing stochastic variability
    for ensemble simulations.
    
    Mathematical Expression:
        W(t+Deltat) = f(W(t), theta, omega(t)) 
        where f captures temporal autocorrelation structure
        
    References:
        - Wilks, D.S. (2006). Statistical Methods in the Atmospheric Sciences.
        - Katz, R.W. (1996). Use of conditional stochastic models to generate
          climate change scenarios. Climate Change 32, 237-255.
        - Kilsby, C.G. et al. (2007). A daily weather generator. J. Hydrol.
          344, 1-10.
    """
    
    @abstractmethod
    def generate_weather(self,
                        start_time: str,
                        end_time: str, 
                        spatial_grid: torch.Tensor,
                        climate_params: Dict[str, torch.Tensor],
                        seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Generate synthetic weather time series.
        
        Parameters:
            start_time (str): ISO format start time
            end_time (str): ISO format end time
            spatial_grid (torch.Tensor): Spatial coordinate grid
            climate_params (Dict[str, torch.Tensor]): Climatological parameters
            seed (Optional[int]): Random seed
            
        Returns:
            Dict[str, torch.Tensor]: Generated weather variables
                Format: {cf_var_name: tensor[time, spatial_dims...]}
        """


class ProcessUncertainty(StochasticProcess):
    """
    Abstract base class for process-level uncertainty.
    
    Implements stochastic processes within biogeochemical models to account
    for unresolved sub-grid variability, measurement uncertainty, and
    model structural uncertainty.
    
    Mathematical Expression:
        dX = f(X, t)dt + g(X, t)dW
        where f is deterministic drift, g is stochastic diffusion
        
    References:
        - Gardiner, C. (2009). Stochastic Methods. Springer.
        - Gillespie, D.T. (1976). A general method for numerically simulating
          stochastic chemical reactions. J. Comput. Phys. 22, 403-434.
    """
    
    @abstractmethod
    def evolve_state(self,
                    initial_state: torch.Tensor,
                    time_steps: torch.Tensor,
                    process_params: Dict[str, torch.Tensor],
                    seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Evolve state through stochastic process.
        
        Parameters:
            initial_state (torch.Tensor): Initial state variables
            time_steps (torch.Tensor): Time discretization
            process_params (Dict[str, torch.Tensor]): Process parameters
            seed (Optional[int]): Random seed
            
        Returns:
            Dict[str, torch.Tensor]: State evolution trajectory
                Format: {state_var: tensor[time_steps, spatial_dims...]}
        """
