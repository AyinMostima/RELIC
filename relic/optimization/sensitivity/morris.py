"""
Morris Screening Method for Global Sensitivity Analysis

This module implements the Morris One-At-A-Time (OAT) screening method for 
identifying the most influential parameters in biogeochemical models. The method
is particularly useful for preliminary parameter screening in high-dimensional 
parameter spaces.

Scientific Background:
    The Morris method uses a one-at-a-time design to explore parameter space
    efficiently. It estimates two sensitivity measures: mu* (mean absolute 
    elementary effect) and sigma (standard deviation of elementary effects).
    Parameters with high mu* have large overall influence, while high sigma 
    indicates non-linear effects or interactions.

References:
    - Morris, M.D. (1991). Factorial sampling plans for preliminary computational experiments
    - Campolongo, F. et al. (2007). An effective screening design for sensitivity analysis
    - Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer
    - Ruano, M.V. et al. (2012). An improved sampling strategy based on trajectory design

Mathematical Expression:
    Elementary Effect: EE_i = [f(x + Deltae_i) - f(x)] / Delta
    where e_i is unit vector for parameter i, Delta is step size
    
    Morris Statistics:
    - mu_i = (1/r) Sigma_j EE_{i,j} (mean elementary effect)
    - mu*_i = (1/r) Sigma_j |EE_{i,j}| (mean absolute elementary effect)  
    - sigma_i = sqrt[(1/(r-1)) Sigma_j (EE_{i,j} - mu_i)^2] (standard deviation)
    
    Where r is number of trajectories
"""

from typing import Dict, List, Optional, Callable
import numpy as np
import logging
from dataclasses import dataclass, field
import time

from .base import (
    SensitivityAnalyzer,
    SensitivityConfig,
    SensitivityResult,
    ParameterRange,
    SensitivityMethod
)
from ...parameters import UnifiedParameterManager, get_global_manager

logger = logging.getLogger(__name__)


@dataclass
class MorrisConfig(SensitivityConfig):
    """Extended configuration specific to Morris method"""
    # The method is fixed by the class (not a constructor argument).
    method: SensitivityMethod = field(default=SensitivityMethod.MORRIS, init=False)
    # Morris-specific parameters. Trajectories are random one-at-a-time
    # designs on the p-level grid (Morris 1991); no optimised trajectory
    # selection (Campolongo et al. 2007) is implemented. None reads the YAML
    # value of each setting. A design of r trajectories over k parameters
    # evaluates the model r (k + 1) times (Morris 1991); that count follows
    # from r and k at analysis time, so the configuration keeps no sample size.
    n_trajectories: Optional[int] = None  # Number of trajectories (r)
    n_levels: Optional[int] = None   # Number of levels for each parameter (p)
    grid_jump: Optional[int] = None  # Grid jump size in levels (Delta * (p - 1))

    def __post_init__(self):
        super().__post_init__()
        manager = get_global_manager()
        if self.n_trajectories is None:
            self.n_trajectories = int(manager.get(
                "optimization.sensitivity.morris.n_trajectories", required=True
            ))
        if self.n_levels is None:
            self.n_levels = int(manager.get("optimization.sensitivity.morris.n_levels", required=True))
        if self.grid_jump is None:
            self.grid_jump = int(manager.get("optimization.sensitivity.morris.grid_jump", required=True))
        
        # Validate Morris-specific parameters
        if self.n_trajectories <= 0:
            raise ValueError("n_trajectories must be positive")
        if self.n_levels < 2:
            raise ValueError("n_levels must be at least 2")
        if self.grid_jump <= 0:
            raise ValueError("grid_jump must be positive")
        # Every level l in {0, ..., p-1} has a one-at-a-time step l +/- grid_jump
        # inside the grid only when 2 * grid_jump <= p (Morris 1991 uses
        # grid_jump = p / 2). A larger jump would force clipped steps.
        if 2 * self.grid_jump > self.n_levels:
            raise ValueError(
                f"grid_jump={self.grid_jump} with n_levels={self.n_levels} cannot keep every "
                "step on the grid; require 2 * grid_jump <= n_levels"
            )


class MorrisAnalyzer(SensitivityAnalyzer):
    """
    Morris One-At-A-Time Screening Method Implementation.
    
    The Morris method is an efficient global sensitivity analysis technique
    that uses elementary effects to identify influential parameters. It's
    particularly suitable for models with many parameters where a full
    Sobol analysis would be computationally prohibitive.
    
    Key Features:
    1. Elementary effects calculation across parameter trajectories
    2. Morris statistics (mu, mu*, sigma) for ranking parameters
    3. Optimal trajectory design for improved space filling
    4. Parameter interaction detection capabilities
    5. Efficient sampling for high-dimensional problems
    
    Applications in biogeochemical modeling:
    - Parameter screening for photosynthesis models
    - Identifying key soil decomposition parameters
    - Screening BVOC emission factors
    - Preliminary analysis before detailed calibration
    """
    
    def __init__(
        self,
        parameter_manager: UnifiedParameterManager,
        config: Optional[MorrisConfig] = None
    ):
        """
        Initialize Morris analyzer.
        
        Parameters:
            parameter_manager: Parameter management system
            config: Morris-specific configuration
        """
        if config is None:
            raise TypeError(
                "MorrisAnalyzer needs a MorrisConfig: the aggregation method is chosen by the caller"
            )
        self.morris_config = config
        super().__init__(parameter_manager, self.morris_config)
        
        # Morris-specific state
        self.trajectories = None
        self.elementary_effects = {}
        
        logger.info(f"Morris analyzer initialized with {self.morris_config.n_trajectories} trajectories")
    
    def analyze(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        parameter_ranges: Dict[str, ParameterRange],
        baseline_parameters: Optional[Dict[str, float]] = None
    ) -> SensitivityResult:
        """
        Perform Morris sensitivity analysis.
        
        Parameters:
            model_runner: Function that runs model and returns outputs
            parameter_ranges: Parameter sampling ranges
            baseline_parameters: Baseline parameter values (optional)
            
        Returns:
            Morris sensitivity analysis results
        """
        self.start_time = time.time()
        
        # Validate inputs
        self.validate_parameter_ranges(parameter_ranges)
        
        if len(parameter_ranges) == 0:
            raise ValueError("At least one parameter range must be specified")
        self.require_uniform_ranges(parameter_ranges, "Morris screening")
        
        param_names = sorted(parameter_ranges.keys())
        n_params = len(param_names)
        self.output_order = None
        
        logger.info(f"Starting Morris analysis for {n_params} parameters")
        
        try:
            # 1. Generate Morris trajectories
            trajectories = self._generate_morris_trajectories(parameter_ranges, param_names)
            
            # 2. Evaluate model along trajectories
            trajectory_outputs = self._evaluate_trajectories(
                model_runner, trajectories, param_names, baseline_parameters
            )
            
            # 3. Calculate elementary effects
            elementary_effects = self._calculate_elementary_effects(
                trajectories, trajectory_outputs, parameter_ranges, param_names
            )
            
            # 4. Compute Morris statistics
            morris_stats = self._compute_morris_statistics(elementary_effects)
            
            # 5. Create sensitivity result
            computation_time = time.time() - self.start_time
            result = self._create_morris_result(
                morris_stats, param_names, elementary_effects,
                n_model_evaluations=self.model_evaluations,
                computation_time=computation_time,
            )

            logger.info(f"Morris analysis completed: {self.model_evaluations} evaluations in {computation_time:.2f}s")
            
            return result
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"Morris analysis failed: {e}")
            raise RuntimeError(f"Morris analysis failed: {e}") from e
    
    def _generate_morris_trajectories(
        self,
        parameter_ranges: Dict[str, ParameterRange],
        param_names: List[str]
    ) -> np.ndarray:
        """
        Generate Morris trajectories using optimal design.
        
        Returns array of shape [n_trajectories, n_params+1, n_params]
        Each trajectory has n_params+1 points, changing one parameter at a time.
        """
        n_params = len(param_names)
        n_trajectories = self.morris_config.n_trajectories
        n_levels = self.morris_config.n_levels
        grid_jump = self.morris_config.grid_jump
        
        # Generate trajectories
        trajectories = np.zeros((n_trajectories, n_params + 1, n_params))

        # A local generator seeded from the config keeps the design
        # reproducible without resetting the global numpy random state.
        rng = np.random.default_rng(self.morris_config.random_seed)
        lower = np.array([parameter_ranges[name].lower_bound for name in param_names])
        span = np.array([
            parameter_ranges[name].upper_bound - parameter_ranges[name].lower_bound
            for name in param_names
        ])

        # References: Morris (1991), Technometrics 33, 161-174; Saltelli et al.
        #     (2008), Global Sensitivity Analysis: The Primer, Sect. 3.3.
        # Mathematical expression: x_i = lower_i + span_i * l_i / (p - 1), with
        #     integer level l_i in {0, ..., p - 1}; one step moves one parameter
        #     l_i -> l_i + s * grid_jump, s in {-1, +1}, so
        #     Delta_i = grid_jump * span_i / (p - 1).
        # Parameter meanings: p = n_levels, grid_jump = step in levels,
        #     lower_i, span_i = lower bound and width of the range of parameter i,
        #     s = step direction.
        # Units: x_i and Delta_i carry the units of parameter i; l_i, p and
        #     grid_jump are dimensionless level counts.
        # Domain/constraints: 2 * grid_jump <= p (MorrisConfig), so for every
        #     level either +grid_jump or -grid_jump stays in {0, ..., p - 1};
        #     the random direction is reversed when it would leave the grid,
        #     so no step is clipped and every step changes its parameter.
        # Sign convention: s = +1 increases the parameter, s = -1 decreases it.
        for traj_idx in range(n_trajectories):
            levels = rng.integers(0, n_levels, size=n_params)
            trajectories[traj_idx, 0, :] = lower + span * levels / (n_levels - 1)

            # Generate trajectory by changing one parameter at a time
            for step, param_idx in enumerate(rng.permutation(n_params)):
                direction = int(rng.choice((-1, 1)))
                if not 0 <= levels[param_idx] + direction * grid_jump <= n_levels - 1:
                    direction = -direction
                new_level = levels[param_idx] + direction * grid_jump
                if not 0 <= new_level <= n_levels - 1:
                    raise ValueError(
                        f"grid_jump={grid_jump} cannot move level {levels[param_idx]} "
                        f"inside n_levels={n_levels}"
                    )
                levels[param_idx] = new_level
                trajectories[traj_idx, step + 1, :] = lower + span * levels / (n_levels - 1)
        
        self.trajectories = trajectories
        logger.debug(f"Generated {n_trajectories} Morris trajectories")
        
        return trajectories
    
    def _evaluate_trajectories(
        self,
        model_runner: Callable[[Dict[str, float]], Dict[str, float]],
        trajectories: np.ndarray,
        param_names: List[str],
        baseline_parameters: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate model at all trajectory points.
        
        Returns dictionary mapping output variables to arrays of shape
        [n_trajectories, n_params+1] containing model outputs. Every run
        must return every analysed output as a finite value.
        """
        n_trajectories, n_points, n_params = trajectories.shape
        trajectory_outputs = {}
        
        for traj_idx in range(n_trajectories):
            for point_idx in range(n_points):
                param_dict = self.model_parameters(
                    param_names, trajectories[traj_idx, point_idx], baseline_parameters
                )
                
                # Evaluate model
                try:
                    values = self.collect_model_outputs(model_runner(param_dict))
                    for var_name, value in values.items():
                        if var_name not in trajectory_outputs:
                            trajectory_outputs[var_name] = np.zeros((n_trajectories, n_points))
                        trajectory_outputs[var_name][traj_idx, point_idx] = value
                    
                    self.model_evaluations += 1
                    
                except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                    logger.error(f"Model evaluation failed at trajectory {traj_idx}, point {point_idx}: {e}")
                    raise RuntimeError(
                        f"Model evaluation failed at trajectory {traj_idx}, point {point_idx}"
                    ) from e
        
        logger.debug(f"Evaluated {n_trajectories * n_points} model runs across trajectories")
        return trajectory_outputs
    
    def _calculate_elementary_effects(
        self,
        trajectories: np.ndarray,
        trajectory_outputs: Dict[str, np.ndarray],
        parameter_ranges: Dict[str, ParameterRange],
        param_names: List[str]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate elementary effects for each parameter and output variable.
        
        Returns nested dictionary: {output_var: {param_name: effects_array}}
        """
        n_trajectories, n_points, n_params = trajectories.shape
        elementary_effects = {}

        # References: Morris (1991), Technometrics 33, 161-174.
        # Mathematical expression: EE_i = [f(x + Delta_i e_i) - f(x)] / Delta_i,
        #     evaluated at the one step of each trajectory that moves parameter i.
        # Parameter meanings: f = analysed model output, x = trajectory point
        #     before the step, e_i = unit vector of parameter i, Delta_i = signed
        #     change of parameter i on that step.
        # Units: EE_i in (output units) / (units of parameter i).
        # Domain/constraints: the grid design changes each parameter on exactly
        #     one step per trajectory with Delta_i != 0; a trajectory without such
        #     a step is an error, never an effect of zero.
        # Sign convention: EE_i > 0 when the output increases with parameter i.
        for var_name, outputs in trajectory_outputs.items():
            elementary_effects[var_name] = {}
            
            for param_idx, param_name in enumerate(param_names):
                effects = np.zeros(n_trajectories)
                
                for traj_idx in range(n_trajectories):
                    changed = np.flatnonzero(
                        np.diff(trajectories[traj_idx, :, param_idx]) != 0.0
                    )
                    if changed.size != 1:
                        raise RuntimeError(
                            f"Trajectory {traj_idx} changes parameter '{param_name}' on "
                            f"{changed.size} steps; the Morris design requires exactly one"
                        )
                    step = int(changed[0])
                    delta_param = (
                        trajectories[traj_idx, step + 1, param_idx]
                        - trajectories[traj_idx, step, param_idx]
                    )
                    effects[traj_idx] = (
                        outputs[traj_idx, step + 1] - outputs[traj_idx, step]
                    ) / delta_param
                
                elementary_effects[var_name][param_name] = effects
        
        self.elementary_effects = elementary_effects
        return elementary_effects
    
    def _compute_morris_statistics(
        self,
        elementary_effects: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute Morris statistics (mu, mu*, sigma) from elementary effects.
        
        Returns: {output_var: {param_name: {stat_name: value}}}
        """
        morris_stats = {}
        
        for var_name, param_effects in elementary_effects.items():
            morris_stats[var_name] = {}
            
            for param_name, effects in param_effects.items():
                # A non-finite effect is an error, never dropped from the
                # statistics (dropping would silently reduce r).
                if not np.all(np.isfinite(effects)):
                    raise ValueError(
                        f"Non-finite elementary effects for {param_name} -> {var_name}"
                    )

                # References: Morris (1991); Campolongo et al. (2007), Environ.
                #     Model. Softw. 22, 1509-1518 (mu*).
                # Mathematical expression: mu_i = mean_j EE_ij, mu*_i = mean_j |EE_ij|,
                #     sigma_i = sqrt(sum_j (EE_ij - mu_i)^2 / (r - 1)).
                # Parameter meanings: EE_ij = elementary effect of parameter i on
                #     trajectory j, r = number of trajectories.
                # Units: all three in (output units) / (units of parameter i).
                # Domain/constraints: sigma_i needs r >= 2; with r = 1 it is
                #     undefined and reported as NaN.
                # Sign convention: mu_i keeps the sign of the effects; mu*_i and
                #     sigma_i are non-negative.
                mu = np.mean(effects)
                mu_star = np.mean(np.abs(effects))
                sigma = np.std(effects, ddof=1) if len(effects) > 1 else float("nan")
                
                morris_stats[var_name][param_name] = {
                    'mu': float(mu),
                    'mu_star': float(mu_star),
                    'sigma': float(sigma)
                }
        
        return morris_stats
    
    def _create_morris_result(
        self,
        morris_stats: Dict[str, Dict[str, Dict[str, float]]],
        param_names: List[str],
        elementary_effects: Dict[str, Dict[str, np.ndarray]],
        *,
        n_model_evaluations: int,
        computation_time: float,
    ) -> SensitivityResult:
        """Create SensitivityResult from Morris statistics"""
        
        # Extract first-order indices (use mu* as first-order measure)
        first_order_indices = {}
        morris_mu = {}
        morris_sigma = {}
        parameter_rankings = {}
        
        for var_name, param_stats in morris_stats.items():
            first_order_indices[var_name] = {}
            morris_mu[var_name] = {}
            morris_sigma[var_name] = {}
            
            # Extract statistics
            for param_name, stats in param_stats.items():
                first_order_indices[var_name][param_name] = stats['mu_star']
                morris_mu[var_name][param_name] = stats['mu']
                morris_sigma[var_name][param_name] = stats['sigma']
            
            # Create parameter ranking based on mu*
            sorted_params = sorted(
                param_stats.items(),
                key=lambda x: abs(x[1]['mu_star']),
                reverse=True
            )
            parameter_rankings[var_name] = [param_name for param_name, _ in sorted_params]
        
        # Create result
        result = SensitivityResult(
            method="Morris",
            parameter_names=param_names,
            output_variables=list(morris_stats.keys()),
            first_order_indices=first_order_indices,
            parameter_rankings=parameter_rankings,
            morris_mu=morris_mu,
            morris_sigma=morris_sigma,
            n_model_evaluations=n_model_evaluations,
            computation_time=computation_time,
        )
        
        return result
