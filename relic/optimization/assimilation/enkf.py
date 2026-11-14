"""
Ensemble Kalman Filter Implementation

This module implements the Ensemble Kalman Filter (EnKF) for data assimilation
in biogeochemical models. The implementation follows the standard EnKF algorithm
with options for inflation, localization, and adaptive techniques.

Scientific Background:
    The EnKF approximates the forecast error covariance using a Monte Carlo ensemble.
    Each ensemble member represents a possible state trajectory, and their spread
    provides an estimate of uncertainty. The method is well-suited for nonlinear
    models and can handle large state spaces efficiently.

References:
    - Evensen, G. (2003). The Ensemble Kalman Filter: theoretical formulation
    - Burgers, G. et al. (1998). Analysis scheme in the ensemble Kalman filter  
    - Anderson, J.L. (2001). An Ensemble Adjustment Kalman Filter for Data Assimilation
    - Hamill, T.M. (2006). Ensemble-based atmospheric data assimilation

Mathematical Expression:
    Forecast step: x_f^i(t+1) = M[x_a^i(t)]  for i=1...N_ens
    
    Analysis step:
    1. Ensemble mean: x?_f = (1/N_ens)sumx_f^i
    2. Anomaly matrix: A_f = [x_f^1-x?_f, ..., x_f^N_ens-x?_f]
    3. Forecast covariance: P_f ~= (1/(N_ens-1))A_f A_f^T
    4. Kalman gain: K = P_f H^T(H P_f H^T + R)^{-1}
    5. Analysis: x_a^i = x_f^i + K(y^i - H x_f^i)
    
    Where y^i are perturbed observations: y^i = y + epsilon^i, epsilon^i ~ N(0,R)
"""

from typing import Dict, Optional, Any
import torch
from torch import Tensor
import logging
import time
from dataclasses import dataclass, field

from .base import (
    Assimilator,
    AssimilationConfig,
    AssimilationMethod,
    AssimilationResult,
    ObservationSet,
    ObservationOperator
)
from relic.core.simulator import RELICSimulator
from relic.core.state import StateManager  
from relic.parameters import UnifiedParameterManager, get_global_manager

logger = logging.getLogger(__name__)


@dataclass
class EnKFConfig(AssimilationConfig):
    """Extended configuration specific to EnKF.

    The method is fixed to ENSEMBLE_KALMAN_FILTER by the class (it is not a
    constructor argument). initial_perturbation_std is required: it is an
    absolute standard deviation in the units of the state, which has no
    universal value. parameter_perturbation_std, covariance_inflation and
    regularization left at None read optimization.assimilation.enkf.* from
    the parameter database.
    Covariance localization, adaptive inflation, SVD truncation, a variance
    floor and observation batching are not implemented; their former fields
    were accepted and silently ignored and are no longer offered.
    efficient_gain selects the ensemble-space gain, the only implemented
    gain: the explicit-covariance variant replaced the observation operator
    Jacobian by an identity matrix and is refused.
    """
    method: AssimilationMethod = field(
        default=AssimilationMethod.ENSEMBLE_KALMAN_FILTER, init=False
    )

    # Ensemble generation
    initial_perturbation_std: float = field(kw_only=True)
    parameter_perturbation: bool = False
    parameter_perturbation_std: Optional[float] = None

    # Covariance treatment (multiplicative inflation of the analysis anomalies)
    covariance_inflation: Optional[float] = None

    # Numerical stability: r in (HPH^T + R + r I)
    regularization: Optional[float] = None
    
    # Gain formulation
    efficient_gain: bool = True
    
    def __post_init__(self):
        """Validate EnKF configuration parameters"""
        super().__post_init__()
        manager = get_global_manager()
        for name in ("parameter_perturbation_std", "covariance_inflation", "regularization"):
            if getattr(self, name) is None:
                setattr(self, name, float(manager.get(
                    f"optimization.assimilation.enkf.{name}", required=True
                )))
        if self.ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        if self.covariance_inflation <= 0:
            raise ValueError("covariance_inflation must be positive")
        if not self.efficient_gain:
            raise NotImplementedError(
                "efficient_gain=False (explicit P_f gain) needs the observation "
                "operator Jacobian, which is not implemented; the ensemble-space "
                "gain is exact for the ensemble covariance"
            )
    

class EnsembleKalmanFilter(Assimilator):
    """
    Ensemble Kalman Filter implementation for biogeochemical data assimilation.
    
    This implementation provides:
    1. Standard EnKF with perturbed observations
    2. Ensemble Transform Kalman Filter (ETKF) option
    3. Covariance inflation and localization
    4. Adaptive techniques for ensemble management
    5. Efficient computation for large state spaces
    
    The filter can assimilate various observation types including:
    - Direct state observations (GPP, ET, NEE)
    - Remote sensing data (LAI, fAPAR)
    - Flux tower measurements
    - Soil moisture and temperature profiles
    """
    
    def __init__(
        self,
        simulator: RELICSimulator,
        state_manager: StateManager,
        parameter_manager: UnifiedParameterManager,
        config: Optional[EnKFConfig] = None
    ):
        """
        Initialize Ensemble Kalman Filter.
        
        Parameters:
            simulator: Model simulator for ensemble forecasts
            state_manager: State management system
            parameter_manager: Parameter management system  
            config: EnKF-specific configuration
        """
        if config is None:
            raise TypeError(
                "EnsembleKalmanFilter needs an EnKFConfig: initial_perturbation_std "
                "(in the units of the state) has no default"
            )
        self.enkf_config = config
        super().__init__(simulator, state_manager, parameter_manager, self.enkf_config)
        
        # Ensemble state storage
        self.ensemble_states: Optional[Dict[str, Tensor]] = None
        self.ensemble_parameters: Optional[Dict[str, Tensor]] = None
        
        # Analysis statistics
        self.analysis_increment_stats = []
        self.innovation_stats = []
        
        logger.info(f"EnKF initialized with {self.enkf_config.ensemble_size} ensemble members")
    
    def initialize_ensemble(
        self, 
        initial_state: Dict[str, Tensor],
        initial_parameters: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Initialize ensemble with perturbed initial conditions.
        
        Parameters:
            initial_state: Mean initial state
            initial_parameters: Mean initial parameters (optional)
        """
        ensemble_size = self.enkf_config.ensemble_size
        
        # Initialize state ensemble
        self.ensemble_states = {}
        for var_name, mean_state in initial_state.items():
            # Create ensemble dimension
            ensemble_shape = (ensemble_size,) + mean_state.shape
            ensemble_var = torch.zeros(ensemble_shape, dtype=self.dtype, device=self.device)
            
            # Add perturbations
            for i in range(ensemble_size):
                perturbation = torch.randn_like(mean_state) * self.enkf_config.initial_perturbation_std
                ensemble_var[i] = mean_state + perturbation
                
            self.ensemble_states[var_name] = ensemble_var
        
        # Initialize parameter ensemble if requested
        if self.enkf_config.parameter_perturbation and initial_parameters:
            self.ensemble_parameters = {}
            for param_name, mean_value in initial_parameters.items():
                perturbations = torch.randn(ensemble_size, dtype=self.dtype, device=self.device)
                perturbations *= self.enkf_config.parameter_perturbation_std * abs(mean_value)
                self.ensemble_parameters[param_name] = mean_value + perturbations
        
        logger.info(f"Ensemble initialized with {len(self.ensemble_states)} state variables")
    
    def forecast_ensemble(self, forecast_length: float) -> None:
        """
        Run forecast for all ensemble members.
        
        Parameters:
            forecast_length: Length of forecast in simulation time units
        """
        if self.ensemble_states is None:
            raise RuntimeError("Ensemble not initialized. Call initialize_ensemble() first.")
        
        # The member forecast is not bound to the canonical runtime: the
        # former loop called RELICSimulator.run(steps=..., save_interval=...),
        # a signature the simulator does not have, swallowed the TypeError
        # and kept every member's previous state as its 'forecast'.
        raise NotImplementedError(
            "EnKF ensemble forecast is not bound to the canonical simulator "
            "runtime ([N] runtime_unbound: per-member StateManager, "
            "ParameterManager and RNG isolation and the forecast time window "
            "are not implemented); forecast the members through the "
            "simulator and pass the ensemble states to initialize_ensemble"
        )
    
    def assimilate(
        self,
        forecast_state: Dict[str, Tensor],
        observations: ObservationSet,
        observation_operator: ObservationOperator,
        background_covariance: Optional[Tensor] = None
    ) -> AssimilationResult:
        """
        Perform EnKF analysis step.
        
        Parameters:
            forecast_state: Prior ensemble mean (used for validation)
            observations: Observations to assimilate
            observation_operator: H operator
            background_covariance: Not used in EnKF (estimated from ensemble)
            
        Returns:
            Analysis result with updated ensemble and diagnostics
        """
        start_time = time.time()
        
        if self.ensemble_states is None:
            raise RuntimeError("Ensemble not initialized")
        
        # Validate observations
        valid_observations = self.validate_observations(observations)
        if len(valid_observations) == 0:
            logger.warning("No valid observations - returning forecast")
            return self._create_forecast_result(valid_observations)
        
        try:
            # 1. Compute ensemble mean and anomalies
            ensemble_mean = self._compute_ensemble_mean()
            anomaly_matrix = self._compute_anomaly_matrix(ensemble_mean)
            
            # 2. Apply observation operator to ensemble
            predicted_observations = self._apply_observation_operator_to_ensemble(observation_operator)
            
            # 3. Compute innovation statistics
            mean_predicted = torch.mean(predicted_observations, dim=0)
            obs_vector = valid_observations.get_observation_vector(self.device).to(
                dtype=mean_predicted.dtype
            )
            innovation = obs_vector - mean_predicted
            
            # 4. Compute the ensemble-space Kalman gain (the only implemented gain)
            kalman_gain = self._compute_efficient_kalman_gain(
                anomaly_matrix, predicted_observations, valid_observations
            )
            
            # 5. Update ensemble members
            analysis_ensemble = self._update_ensemble_members(
                kalman_gain, valid_observations, predicted_observations
            )
            
            # 6. Apply inflation if configured
            if self.enkf_config.covariance_inflation > 1.0:
                analysis_ensemble = self._apply_inflation(analysis_ensemble)
            
            # 7. Update stored ensemble
            self.ensemble_states = analysis_ensemble
            
            # 8. Compute analysis mean and diagnostics
            analysis_mean = self._compute_ensemble_mean()
            
            # Create result
            result = AssimilationResult(
                analysis_mean=analysis_mean,
                forecast_mean=ensemble_mean,
                innovation=innovation,
                rms_innovation=torch.sqrt(torch.mean(innovation**2)).item(),
                assimilation_time=valid_observations.time,
                computational_cost=time.time() - start_time,
                observations_assimilated=len(valid_observations),
                observations_rejected=len(observations) - len(valid_observations)
            )
            
            # Add EnKF-specific diagnostics
            result.metadata.update({
                'ensemble_size': self.enkf_config.ensemble_size,
                'inflation_factor': self.enkf_config.covariance_inflation,
            })
            
            # Update statistics
            self.cycle_count += 1
            self.total_observations += len(valid_observations)
            self.total_computational_cost += result.computational_cost
            
            logger.info(f"EnKF analysis completed: {len(valid_observations)} obs, "
                       f"RMS innovation: {result.rms_innovation:.4f}")
            
            return result
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"EnKF analysis failed: {e}")
            raise RuntimeError(f"EnKF analysis failed: {e}") from e
    
    def _compute_ensemble_mean(self) -> Dict[str, Tensor]:
        """Compute ensemble mean for all state variables"""
        ensemble_mean = {}
        for var_name, ensemble in self.ensemble_states.items():
            ensemble_mean[var_name] = torch.mean(ensemble, dim=0)
        return ensemble_mean
    
    def _compute_anomaly_matrix(self, ensemble_mean: Dict[str, Tensor]) -> Tensor:
        """
        Compute anomaly matrix A = [x1-x?, x2-x?, ..., xN-x?]
        
        Returns flattened anomaly matrix for efficient linear algebra
        """
        anomalies = []
        ensemble_size = self.enkf_config.ensemble_size
        
        for i in range(ensemble_size):
            member_anomaly = []
            for var_name in sorted(self.ensemble_states.keys()):
                ensemble = self.ensemble_states[var_name]
                mean_val = ensemble_mean[var_name]
                anomaly = (ensemble[i] - mean_val).flatten()
                member_anomaly.append(anomaly)
            
            # Concatenate all variables for this member
            full_anomaly = torch.cat(member_anomaly)
            anomalies.append(full_anomaly)
        
        # Stack to form anomaly matrix [state_size x ensemble_size]
        anomaly_matrix = torch.stack(anomalies, dim=1)
        return anomaly_matrix
    
    def _apply_observation_operator_to_ensemble(
        self, 
        observation_operator: ObservationOperator
    ) -> Tensor:
        """Apply observation operator to each ensemble member"""
        ensemble_size = self.enkf_config.ensemble_size
        predicted_observations = []
        
        for i in range(ensemble_size):
            # Extract member state
            member_state = {var: ensemble[i] for var, ensemble in self.ensemble_states.items()}
            
            # Apply observation operator
            predicted_obs = observation_operator.forward(member_state)
            predicted_observations.append(predicted_obs)
        
        # Stack predictions [ensemble_size x obs_size]
        return torch.stack(predicted_observations, dim=0)
    
    def _compute_efficient_kalman_gain(
        self,
        anomaly_matrix: Tensor,
        predicted_observations: Tensor,
        observations: ObservationSet
    ) -> Tensor:
        """
        Compute Kalman gain using efficient ensemble space formulation.
        
        This avoids explicit computation of P_f which can be very large.
        Uses the Sherman-Morrison-Woodbury formula.
        """
        ensemble_size = self.enkf_config.ensemble_size
        
        dtype = anomaly_matrix.dtype

        # Observation error covariance
        R = observations.get_error_covariance_matrix(self.device).to(dtype=dtype)
        
        # Predicted observation anomalies
        pred_obs_mean = torch.mean(predicted_observations, dim=0)
        pred_obs_anomalies = predicted_observations - pred_obs_mean.unsqueeze(0)
        
        # Compute (N-1)^{-1} H A A^T H^T + R
        # where A is anomaly matrix and N is ensemble size
        factor = 1.0 / (ensemble_size - 1)
        HAAh = factor * torch.mm(pred_obs_anomalies.T, pred_obs_anomalies)
        innovation_covariance = HAAh + R
        
        # Add regularization for numerical stability
        if self.enkf_config.regularization > 0:
            innovation_covariance += self.enkf_config.regularization * torch.eye(
                innovation_covariance.shape[0], device=self.device, dtype=dtype
            )
        
        # References: Evensen (2003), Ocean Dynamics 53, 343-367.
        # Mathematical expression:
        #   K = (N-1)^-1 A (HA)^T S^-1, S = (N-1)^-1 (HA)^T(HA) + R + r I,
        #   solved as S^T K^T = ((N-1)^-1 A (HA)^T)^T.
        # Parameter meanings: A state anomalies, HA predicted-observation
        #   anomalies, R observation error covariance, r the configured
        #   regularization, N ensemble size.
        # Units: K in state units per observation unit; S in squared
        #   observation units.
        # Domain/constraints: S must be nonsingular; a singular S is an
        #   error, never replaced by a pseudo-inverse gain.
        # Sign convention: K multiplies the innovation y - H(x).
        Ah_term = factor * torch.mm(anomaly_matrix, pred_obs_anomalies)
        try:
            kalman_gain = torch.linalg.solve(innovation_covariance.T, Ah_term.T).T
        except torch.linalg.LinAlgError as err:
            raise RuntimeError(
                "EnKF innovation covariance is singular; increase the "
                "observation error or the declared regularization"
            ) from err
        
        return kalman_gain
    
    def _update_ensemble_members(
        self,
        kalman_gain: Tensor,
        observations: ObservationSet,
        predicted_observations: Tensor
    ) -> Dict[str, Tensor]:
        """Update each ensemble member using perturbed observations"""
        ensemble_size = self.enkf_config.ensemble_size
        dtype = predicted_observations.dtype
        obs_vector = observations.get_observation_vector(self.device).to(dtype=dtype)
        R = observations.get_error_covariance_matrix(self.device).to(dtype=dtype)
        
        # Generate perturbed observations
        perturbed_obs = []
        for i in range(ensemble_size):
            noise = torch.mm(
                torch.linalg.cholesky(R),
                torch.randn(len(observations), 1, device=self.device, dtype=dtype),
            )
            perturbed_obs.append(obs_vector + noise.flatten())
        
        # Update each ensemble member
        updated_ensemble = {}
        for var_name in self.ensemble_states.keys():
            updated_ensemble[var_name] = torch.zeros_like(self.ensemble_states[var_name])
        
        # Flatten ensemble states for linear algebra
        state_start_indices = {}
        current_idx = 0
        for var_name in sorted(self.ensemble_states.keys()):
            state_start_indices[var_name] = current_idx
            current_idx += self.ensemble_states[var_name][0].numel()
        
        for i in range(ensemble_size):
            # Innovation for this member
            innovation = perturbed_obs[i] - predicted_observations[i]
            
            # Apply Kalman gain
            state_increment = torch.mv(kalman_gain, innovation)
            
            # Distribute increment back to state variables
            for var_name in sorted(self.ensemble_states.keys()):
                start_idx = state_start_indices[var_name]
                var_shape = self.ensemble_states[var_name][0].shape
                var_size = var_shape.numel()
                
                # Extract increment for this variable
                var_increment = state_increment[start_idx:start_idx + var_size]
                var_increment = var_increment.reshape(var_shape)
                
                # Update ensemble member
                updated_ensemble[var_name][i] = self.ensemble_states[var_name][i] + var_increment
        
        return updated_ensemble
    
    def _apply_inflation(self, ensemble: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Apply covariance inflation to ensemble"""
        inflation_factor = self.enkf_config.covariance_inflation
        
        inflated_ensemble = {}
        for var_name, var_ensemble in ensemble.items():
            # Compute ensemble mean
            ensemble_mean = torch.mean(var_ensemble, dim=0)
            
            # Inflate anomalies
            anomalies = var_ensemble - ensemble_mean.unsqueeze(0)
            inflated_anomalies = anomalies * inflation_factor
            
            # Reconstruct ensemble
            inflated_ensemble[var_name] = ensemble_mean.unsqueeze(0) + inflated_anomalies
        
        return inflated_ensemble
    
    def _create_forecast_result(self, observations: ObservationSet) -> AssimilationResult:
        """Create result when no observations are assimilated"""
        ensemble_mean = self._compute_ensemble_mean()
        
        return AssimilationResult(
            analysis_mean=ensemble_mean,
            forecast_mean=ensemble_mean,
            # No observation was assimilated: the RMS innovation is undefined.
            rms_innovation=float("nan"),
            assimilation_time=observations.time,
            computational_cost=0.0,
            observations_assimilated=0,
            observations_rejected=len(observations)
        )
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble spread and other statistics"""
        if self.ensemble_states is None:
            return {}
        
        statistics = {}
        ensemble_mean = self._compute_ensemble_mean()
        
        for var_name, ensemble in self.ensemble_states.items():
            mean_val = ensemble_mean[var_name]
            variance = torch.var(ensemble, dim=0)
            spread = torch.sqrt(torch.mean(variance))
            
            statistics[var_name] = {
                'mean': mean_val.mean().item(),
                'ensemble_spread': spread.item(),
                'min': torch.min(ensemble).item(),
                'max': torch.max(ensemble).item()
            }
        
        return statistics
