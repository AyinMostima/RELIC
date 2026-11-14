"""
Validation Metrics and Proper Scoring Rules for RELIC

This module implements strictly proper scoring rules for probabilistic forecast
evaluation, including PIT, CRPS, Brier scores, and quantile scores. All metrics
follow Gneiting & Raftery (2007) guidelines for unbiased forecast assessment.

Key Features:
    - Strictly proper scoring rules (no hedging incentives)
    - Comprehensive probabilistic forecast evaluation
    - Statistical consistency tests (PIT, rank histograms) 
    - Coverage probability assessment
    - CF-1.8 compliant output for benchmarking
    - Efficient computation for ensemble forecasts

Mathematical Framework:
    Proper scoring rule: S(F, y) where F is forecast distribution, y is observation
    Propriety: E[S(F, Y)] <= E[S(G, Y)] if and only if F = G (true distribution)
    
    Core Metrics:
        CRPS: S_CRPS(F,y) = integral(F(t) - H(t-y))^2 dt
        Brier: S_B(p_u,y,u) = (p_u - I_{y>u})^2  
        Quantile: S_Q(F,y,tau) = tau(y-q_tau)I_{y>q_tau} + (tau-1)(y-q_tau)I_{y<=q_tau}
        PIT: z = F(y) should be ~ Uniform(0,1) for calibrated forecasts

References:
    - Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules,
      prediction, and estimation. J. Am. Stat. Assoc. 102, 359-378.
    - Matheson, J.E. & Winkler, R.L. (1976). Scoring rules for continuous
      probability distributions. Management Science 22, 1087-1096.
    - Hersbach, H. (2000). Decomposition of the continuous ranked probability
      score for ensemble forecasts. Weather Forecast. 15, 559-570.
    - Wilks, D.S. (2011). Statistical Methods in the Atmospheric Sciences.
      Academic Press, 3rd edition.

Authors: Yuchong Long
"""

from typing import Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import math
import logging

import torch
from torch import Tensor
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox

from .category import ValidationMetrics, ProcessMetadata
from .ensemble_runner import quantile_label
from relic.parameters import get_global_manager

logger = logging.getLogger(__name__)


def _validate_finite_1d(name: str, values: Tensor) -> None:
    """Validate a non-empty finite one-dimensional tensor."""
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if values.dim() != 1:
        raise ValueError(f"{name} must be 1D, got shape {values.shape}")
    if values.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite values")


def _validate_same_shape(reference_name: str,
                         reference: Tensor,
                         value_name: str,
                         values: Tensor) -> None:
    """Validate that two tensors have the same shape."""
    if values.shape != reference.shape:
        raise ValueError(
            f"{reference_name} and {value_name} must have the same shape, "
            f"got {reference.shape} and {values.shape}"
        )


@dataclass 
class StochasticValidationResult:
    """
    Container for validation results with statistical summaries.
    
    Parameters:
        score (float): Primary score value (lower is better for proper scores)
        decomposition (Dict[str, float]): Score decomposition (reliability, resolution, uncertainty)
        pvalue (Optional[float]): Statistical significance (for consistency tests)
        confidence_interval (Optional[Tuple[float, float]]): Bootstrap confidence bounds
        sample_size (int): Number of forecast-observation pairs
        metadata (Dict[str, Any]): Additional diagnostic information
        
    Mathematical Expression:
        For CRPS decomposition: CRPS = Reliability - Resolution + Uncertainty
        For Brier decomposition: BS = Reliability - Resolution + Uncertainty
        
    References:
        - Murphy, A.H. (1973). A new vector partition of the probability score.
          J. Appl. Meteorol. 12, 595-600.
    """
    score: float
    decomposition: Dict[str, float]
    pvalue: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    sample_size: int = field(kw_only=True)
    metadata: Dict[str, Any] = field(default_factory=dict)


ValidationResult = StochasticValidationResult


class PITValidator(ValidationMetrics):
    """
    Probability Integral Transform (PIT) statistical consistency test.
    
    Tests whether forecast distributions are statistically consistent with
    observations by examining the distribution of PIT values z = F(y).
    For calibrated forecasts, PIT values should follow Uniform(0,1).
    
    Mathematical Expression:
        PIT value: z_i = F_i(y_i) where F_i is forecast CDF, y_i is observation
        Null hypothesis: {z_i} ~ i.i.d. Uniform(0,1)
        Alternative: Forecasts are miscalibrated (biased, over/under-dispersed)
        
    Tests Applied:
        - Kolmogorov-Smirnov test for uniformity
        - Cramer-von Mises test for uniformity
        - Chi-squared test for histogram bins
        - Ljung-Box test for serial independence
        
    References:
        - Diebold, F.X. et al. (1998). Evaluating density forecasts with applications
          to financial risk management. Int. Econ. Rev. 39, 863-883.
        - Gneiting, T. et al. (2007). Probabilistic forecasts, calibration and sharpness.
          J. Roy. Stat. Soc. B 69, 243-268.
    """
    
    def __init__(self, alpha: Optional[float] = None):
        """
        Initialize PIT validator.

        Parameters:
            alpha (Optional[float]): Significance level for statistical
                tests; None reads validation.stochastic_scores.pit_significance_level
        """
        if alpha is None:
            alpha = float(get_global_manager().get(
                "validation.stochastic_scores.pit_significance_level", required=True
            ))
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and strictly between 0 and 1")
        self.alpha = alpha
        
        # ValidationMetrics is an ABC with no __init__ method
        self.metadata = ProcessMetadata(
            name="PITValidator",
            description="Probability Integral Transform consistency test",
            units={"pit_values": "dimensionless", "pvalue": "dimensionless"},
            parameters={"alpha": alpha},
            references="Diebold et al. (1998), Gneiting et al. (2007)"
        )
        
    def compute_score(self,
                     observations: Tensor,
                     predictions: Union[Tensor, Dict[str, Tensor]], 
                     **kwargs) -> Dict[str, float]:
        """
        Compute PIT values and statistical consistency tests.
        
        Parameters:
            observations (Tensor): Observed values [n_obs]
            predictions (Union[Tensor, Dict]): Ensemble forecasts [n_obs, n_members]
                                              or distribution parameters
            **kwargs: Additional options (test_independence, n_bins)
            
        Returns:
            Dict[str, float]: PIT test results and p-values
            
        Mathematical Expression:
            For ensemble forecast: z_i = (1/M) Sigma_{m=1}^M I(x_m^i <= y_i)
            where x_m^i is m-th ensemble member for i-th forecast
            For parametric: z_i = F_i(y_i | theta_i)
        """
        _validate_finite_1d("Observations", observations)
        
        n_obs = len(observations)
        
        # Compute PIT values
        if isinstance(predictions, Tensor):
            if predictions.dim() != 2:
                raise ValueError(f"Ensemble predictions must be 2D [n_obs, n_members], got {predictions.shape}")
            if predictions.shape[0] != n_obs:
                raise ValueError(f"Prediction/observation size mismatch: {predictions.shape[0]} vs {n_obs}")
            if predictions.shape[1] == 0:
                raise ValueError("Ensemble predictions must contain at least one ensemble member")
            if not bool(torch.isfinite(predictions).all()):
                raise ValueError("Ensemble predictions must contain only finite values")
            
            # Ensemble-based PIT
            pit_values = self._compute_ensemble_pit(observations, predictions)
            pit_method = "empirical_ensemble_cdf"
            pit_is_discrete = True
            ensemble_members = int(predictions.shape[1])
            
        elif isinstance(predictions, dict):
            # Parametric distribution PIT
            pit_values = self._compute_parametric_pit(observations, predictions)
            pit_method = "parametric_cdf"
            pit_is_discrete = False
            ensemble_members = None
        else:
            raise TypeError("Predictions must be Tensor (ensemble) or Dict (parametric)")
        
        # Statistical tests
        results = self._run_consistency_tests(pit_values, **kwargs)
        
        # Add diagnostic information
        results['n_observations'] = n_obs
        results['pit_mean'] = float(pit_values.mean())
        results['pit_std'] = float(pit_values.std(unbiased=False))
        results['pit_method'] = pit_method
        results['pit_is_discrete'] = pit_is_discrete
        results['ensemble_members'] = ensemble_members
        if pit_is_discrete:
            results['uniformity_test_interpretation'] = (
                "diagnostic_only_for_finite_ensemble"
            )
        
        return results
    
    def _compute_ensemble_pit(self, observations: Tensor, ensemble: Tensor) -> Tensor:
        """
        Compute PIT values from ensemble forecasts.
        
        Parameters:
            observations (Tensor): Observations [n_obs]
            ensemble (Tensor): Ensemble forecasts [n_obs, n_members]
            
        Returns:
            Tensor: PIT values [n_obs]
            
        Mathematical Expression:
            z_i = (1/M) Sigma_{m=1}^M I(x_m^i <= y_i)
            where I is indicator function
        """
        # Count ensemble members below/equal to observation
        ensemble.shape[1]
        below_obs = (ensemble <= observations.unsqueeze(1)).float()  # [n_obs, n_members]
        pit_values = below_obs.mean(dim=1)  # [n_obs]
        
        return pit_values
    
    def _compute_parametric_pit(self, observations: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """
        Compute PIT values from parametric distribution.
        
        Parameters:
            observations (Tensor): Observations [n_obs]  
            params (Dict[str, Tensor]): Distribution parameters
            
        Returns:
            Tensor: PIT values [n_obs]
        """
        if 'loc' in params and 'scale' in params:
            # Normal distribution
            from torch.distributions import Normal
            _validate_finite_1d("loc", params['loc'])
            _validate_finite_1d("scale", params['scale'])
            _validate_same_shape("Observations", observations, "loc", params['loc'])
            _validate_same_shape("Observations", observations, "scale", params['scale'])
            if bool((params['scale'] <= 0).any()):
                raise ValueError("scale must be strictly positive")
            dist = Normal(params['loc'], params['scale'])
            pit_values = dist.cdf(observations)
        elif 'low' in params and 'high' in params:
            # Uniform distribution
            from torch.distributions import Uniform
            _validate_finite_1d("low", params['low'])
            _validate_finite_1d("high", params['high'])
            _validate_same_shape("Observations", observations, "low", params['low'])
            _validate_same_shape("Observations", observations, "high", params['high'])
            if bool((params['high'] <= params['low']).any()):
                raise ValueError("high must be greater than low")
            dist = Uniform(params['low'], params['high'])
            pit_values = dist.cdf(observations)
        else:
            raise ValueError("Unsupported parametric distribution")
        
        return pit_values
    
    def _run_consistency_tests(self, pit_values: Tensor, **kwargs) -> Dict[str, float]:
        """
        Run statistical consistency tests on PIT values.
        
        Parameters:
            pit_values (Tensor): PIT values to test [n_obs]
            **kwargs: Test options
            
        Returns:
            Dict[str, float]: Test results and p-values
        """
        results = {}
        pit_numpy = pit_values.detach().cpu().numpy()
        
        # Kolmogorov-Smirnov test for uniformity
        ks_stat, ks_pvalue = stats.kstest(
            pit_numpy,
            'uniform',
            method='asymp',
        )
        results['ks_statistic'] = float(ks_stat)
        results['ks_pvalue'] = float(ks_pvalue)
        results['ks_reject'] = ks_pvalue < self.alpha
        results['ks_method'] = 'asymp'
        
        # Cramer-von Mises has a directly supported continuous uniform null.
        cvm_result = stats.cramervonmises(pit_numpy, 'uniform')
        results['cvm_statistic'] = float(cvm_result.statistic)
        results['cvm_pvalue'] = float(cvm_result.pvalue)
        results['cvm_reject'] = cvm_result.pvalue < self.alpha
        
        # Chi-squared test for histogram uniformity. Without an explicit n_bins
        # the test is optional: it runs with min(20, n // 10) bins when every
        # bin expects at least 5 values (Cochran 1954) and is otherwise absent
        # from the results. An explicit n_bins requests the test: it runs or
        # raises, it is never skipped silently.
        requested_bins = kwargs.get('n_bins')
        n_bins = requested_bins if requested_bins is not None else min(20, len(pit_numpy) // 10)
        chi2_runnable = n_bins > 1 and len(pit_numpy) >= n_bins * 5
        if requested_bins is not None and not chi2_runnable:
            raise ValueError(
                f"chi-squared PIT test with n_bins={requested_bins} needs n_bins > 1 and "
                f"at least 5 values per bin, got {len(pit_numpy)} values"
            )
        if chi2_runnable:
            hist, _ = np.histogram(pit_numpy, bins=n_bins, range=(0, 1))
            expected_freq = len(pit_numpy) / n_bins
            chi2_stat, chi2_pvalue = stats.chisquare(hist, f_exp=expected_freq)
            results['chi2_statistic'] = float(chi2_stat)
            results['chi2_pvalue'] = float(chi2_pvalue)
            results['chi2_reject'] = chi2_pvalue < self.alpha
        
        # Test for serial independence (Ljung-Box test). A requested test
        # runs or raises: too few values or a failing test is an error, never
        # a silently missing result.
        if kwargs.get('test_independence', False):
            if len(pit_numpy) <= 20:
                raise ValueError(
                    "test_independence needs more than 20 PIT values, got "
                    f"{len(pit_numpy)}"
                )
            lags = min(10, len(pit_numpy) // 4)
            lb_result = acorr_ljungbox(pit_numpy, lags=lags, return_df=True)
            results['ljungbox_pvalue'] = float(lb_result['lb_pvalue'].iloc[-1])
            results['ljungbox_reject'] = results['ljungbox_pvalue'] < self.alpha
        
        return results
    
    def diagnostic_plots(self,
                        observations: Tensor,
                        predictions: Union[Tensor, Dict[str, Tensor]],
                        save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate PIT diagnostic plots.
        
        Parameters:
            observations (Tensor): Observed values
            predictions: Forecast ensemble or parameters  
            save_path (Optional[str]): Path to save plots
            
        Returns:
            Dict[str, Any]: Plot data and statistics
        """
        # Compute PIT values
        if isinstance(predictions, Tensor):
            pit_values = self._compute_ensemble_pit(observations, predictions)
        else:
            pit_values = self._compute_parametric_pit(observations, predictions)
        
        pit_numpy = pit_values.detach().cpu().numpy()
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('PIT Diagnostic Plots', fontsize=16)
        
        # 1. PIT histogram
        axes[0, 0].hist(pit_numpy, bins=20, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        axes[0, 0].axhline(y=1.0, color='red', linestyle='--', label='Perfect calibration')
        axes[0, 0].set_xlabel('PIT Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('PIT Histogram')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q plot against uniform
        sorted_pit = np.sort(pit_numpy)
        uniform_quantiles = np.linspace(0, 1, len(pit_numpy))
        axes[0, 1].scatter(uniform_quantiles, sorted_pit, alpha=0.6, s=20)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        axes[0, 1].set_xlabel('Theoretical Uniform Quantiles')
        axes[0, 1].set_ylabel('Sample PIT Quantiles')
        axes[0, 1].set_title('Q-Q Plot vs Uniform')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. PIT time series (if ordered)
        axes[1, 0].plot(pit_numpy, 'b-', alpha=0.7, linewidth=1)
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', label='Expected mean')
        axes[1, 0].fill_between(range(len(pit_numpy)), 0.25, 0.75, alpha=0.2, color='red', label='Central 50%')
        axes[1, 0].set_xlabel('Forecast Index')
        axes[1, 0].set_ylabel('PIT Value')
        axes[1, 0].set_title('PIT Time Series')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. PIT vs observation value (conditional calibration)
        axes[1, 1].scatter(observations.detach().cpu().numpy(), pit_numpy, alpha=0.6, s=20)
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', label='Expected mean')
        axes[1, 1].set_xlabel('Observation Value')
        axes[1, 1].set_ylabel('PIT Value')
        axes[1, 1].set_title('PIT vs Observation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"PIT plots saved to {save_path}")
        
        # Return diagnostic data
        return {
            'pit_values': pit_values,
            'pit_mean': float(pit_numpy.mean()),
            'pit_std': float(pit_numpy.std()),
            'pit_min': float(pit_numpy.min()),
            'pit_max': float(pit_numpy.max()),
            'figure': fig
        }


class CRPSValidator(ValidationMetrics):
    """
    Continuous Ranked Probability Score (CRPS) computation.
    
    CRPS is a strictly proper scoring rule that measures the difference
    between forecast and observation cumulative distribution functions.
    Lower CRPS indicates better forecast skill.
    
    Mathematical Expression:
        CRPS(F,y) = integral_{-inf}^{inf} [F(t) - H(t-y)]^2 dt
        where F(t) is forecast CDF, H(t-y) is Heaviside step function
        
    For ensemble forecasts:
        CRPS = (1/M) Sigma_{i=1}^M |x_i - y| - (1/2M^2) Sigma_{i=1}^M Sigma_{j=1}^M |x_i - x_j|
        where {x_i} are ensemble members
        
    References:
        - Matheson, J.E. & Winkler, R.L. (1976). Scoring rules for continuous
          probability distributions. Management Science 22, 1087-1096.
        - Hersbach, H. (2000). Decomposition of the continuous ranked probability
          score for ensemble forecasts. Weather Forecast. 15, 559-570.
    """
    
    def __init__(self, decompose: bool = True):
        """
        Initialize CRPS validator.
        
        Parameters:
            decompose (bool): Whether to compute reliability-resolution decomposition
        """
        # Store instance variables first
        self.decompose = decompose
        
        # ValidationMetrics is an ABC with no __init__ method
        self.metadata = ProcessMetadata(
            name="CRPSValidator", 
            description="Continuous Ranked Probability Score evaluation",
            units={"crps": "same_as_observations", "components": "same_as_observations"},
            parameters={"decompose": decompose},
            references="Matheson & Winkler (1976), Hersbach (2000)"
        )
        self.decompose = decompose
    
    def compute_score(self,
                     observations: Tensor,
                     predictions: Union[Tensor, Dict[str, Tensor]], 
                     **kwargs) -> Dict[str, float]:
        """
        Compute CRPS score and optional decomposition.
        
        Parameters:
            observations (Tensor): Observed values [n_obs]
            predictions (Union[Tensor, Dict]): Ensemble forecasts [n_obs, n_members]
                                              or distribution parameters
            **kwargs: fair_crps (bool, ensemble forecasts only). Any other
                option is rejected; observation weights are not implemented.

        Returns:
            Dict[str, float]: CRPS score and decomposition components
        """
        unknown = sorted(set(kwargs) - {"fair_crps"})
        if unknown:
            raise TypeError(f"CRPSValidator.compute_score got unsupported options: {unknown}")
        _validate_finite_1d("Observations", observations)

        if isinstance(predictions, Tensor):
            # Ensemble CRPS
            crps_values = self._compute_ensemble_crps(observations, predictions, **kwargs)
        elif isinstance(predictions, dict):
            # Parametric CRPS; the fair (finite-ensemble) correction has no
            # meaning for a closed-form predictive distribution.
            if kwargs.get("fair_crps", False):
                raise ValueError("fair_crps applies to ensemble forecasts, not parametric ones")
            crps_values = self._compute_parametric_crps(observations, predictions)
        else:
            raise TypeError("Predictions must be Tensor (ensemble) or Dict (parametric)")
        
        results = {
            'crps_mean': float(crps_values.mean()),
            'crps_std': float(crps_values.std(unbiased=False)),
            'crps_values': crps_values
        }
        
        # Add decomposition if requested
        if self.decompose and isinstance(predictions, Tensor):
            if kwargs.get('fair_crps', False):
                results['decomposition_available'] = False
                results['decomposition_reason'] = (
                    "standard_empirical_cdf_decomposition_not_valid_for_fair_crps"
                )
            else:
                decomp = self._compute_crps_decomposition(observations, predictions)
                results.update(decomp)
        elif self.decompose:
            # The Hersbach decomposition is defined for ensemble forecasts.
            results['decomposition_available'] = False
            results['decomposition_reason'] = "decomposition_defined_for_ensemble_forecasts_only"

        return results
    
    def _compute_ensemble_crps(self, 
                              observations: Tensor, 
                              ensemble: Tensor,
                              fair_crps: bool = False,
                              **kwargs) -> Tensor:
        """
        Compute CRPS for ensemble forecasts using efficient formula.
        
        Parameters:
            observations (Tensor): Observations [n_obs]
            ensemble (Tensor): Ensemble forecasts [n_obs, n_members]
            fair_crps (bool): Use bias-corrected "fair" CRPS
            
        Returns:
            Tensor: CRPS values for each forecast [n_obs]
            
        Mathematical Expression:
            CRPS = E|X - y| - (1/2)E|X - X'|
            where X, X' are independent copies of forecast distribution
        """
        if ensemble.dim() != 2:
            raise ValueError(
                f"Ensemble predictions must be 2D [n_obs, n_members], got {ensemble.shape}"
            )
        n_obs, n_members = ensemble.shape
        if ensemble.shape[0] != len(observations):
            raise ValueError(f"Size mismatch: {ensemble.shape[0]} vs {len(observations)}")
        if n_obs == 0:
            raise ValueError("Observations and ensemble predictions must be non-empty")
        if n_members == 0:
            raise ValueError("Ensemble predictions must contain at least one ensemble member")
        if not bool(torch.isfinite(ensemble).all()):
            raise ValueError("Ensemble predictions must contain only finite values")
        if fair_crps and n_members < 2:
            raise ValueError("Fair CRPS requires at least two ensemble members")

        # First term: mean absolute error between ensemble and observations
        abs_errors = torch.abs(ensemble - observations.unsqueeze(1))  # [n_obs, n_members]
        first_term = abs_errors.mean(dim=1)  # [n_obs]
        
        # Second term: mean absolute difference within ensemble
        # Compute pairwise differences efficiently
        ensemble_expanded_i = ensemble.unsqueeze(2)  # [n_obs, n_members, 1]
        ensemble_expanded_j = ensemble.unsqueeze(1)  # [n_obs, 1, n_members]
        pairwise_diffs = torch.abs(ensemble_expanded_i - ensemble_expanded_j)  # [n_obs, n_members, n_members]
        
        if fair_crps:
            # Exclude diagonal (same member comparisons) for fair CRPS
            mask = ~torch.eye(n_members, dtype=torch.bool, device=ensemble.device)
            second_term = pairwise_diffs[:, mask].mean(dim=1)  # [n_obs]
        else:
            second_term = pairwise_diffs.mean(dim=(1, 2))  # [n_obs]
        
        crps_values = first_term - 0.5 * second_term
        
        return crps_values
    
    def _compute_parametric_crps(self, observations: Tensor, params: Dict[str, Tensor]) -> Tensor:
        """
        Compute CRPS for parametric distributions with analytical formulas.
        
        Parameters:
            observations (Tensor): Observations [n_obs]
            params (Dict[str, Tensor]): Distribution parameters
            
        Returns:
            Tensor: CRPS values [n_obs]
        """
        if 'loc' in params and 'scale' in params:
            # Normal distribution - analytical CRPS
            _validate_finite_1d("loc", params['loc'])
            _validate_finite_1d("scale", params['scale'])
            _validate_same_shape("Observations", observations, "loc", params['loc'])
            _validate_same_shape("Observations", observations, "scale", params['scale'])
            if bool((params['scale'] <= 0).any()):
                raise ValueError("scale must be strictly positive")
            return self._crps_normal(observations, params['loc'], params['scale'])
        elif 'low' in params and 'high' in params:
            # Uniform distribution - analytical CRPS
            _validate_finite_1d("low", params['low'])
            _validate_finite_1d("high", params['high'])
            _validate_same_shape("Observations", observations, "low", params['low'])
            _validate_same_shape("Observations", observations, "high", params['high'])
            if bool((params['high'] <= params['low']).any()):
                raise ValueError("high must be greater than low")
            return self._crps_uniform(observations, params['low'], params['high'])
        else:
            raise ValueError(
                "Unsupported parametric distribution; expected loc/scale or low/high"
            )
    
    def _crps_normal(self, observations: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
        """
        Analytical CRPS for normal distribution.
        
        Mathematical Expression:
            CRPS = sigma[z*(2Phi(z) - 1) + 2phi(z) - 1/sqrtpi]
            where z = (y - mu)/sigma, Phi is standard normal CDF, phi is standard normal PDF
        """
        standardized = (observations - loc) / scale
        
        from torch.distributions import Normal
        std_normal = Normal(0, 1)
        
        phi_z = std_normal.cdf(standardized)  # Phi(z)
        pdf_z = torch.exp(std_normal.log_prob(standardized))  # phi(z)
        
        crps = scale * (
            standardized * (2 * phi_z - 1) + 
            2 * pdf_z - 
            1 / np.sqrt(np.pi)
        )
        
        return crps
    
    def _crps_uniform(self, observations: Tensor, low: Tensor, high: Tensor) -> Tensor:
        """
        Analytical CRPS for uniform distribution.
        
        Mathematical Expression:
            CRPS = E|X-y| - 0.5 E|X-X'|, X and X' ~ Uniform(a,b)
            For y in [a,b]:
            CRPS = ((y-a)^2 + (b-y)^2) / (2(b-a)) - (b-a)/6
            Outside [a,b], the distance to the nearest bound is added to
            (b-a)/3.
        """
        width = high - low
        inside = (
            (observations - low) ** 2 + (high - observations) ** 2
        ) / (2 * width) - width / 6
        below = low - observations + width / 3
        above = observations - high + width / 3

        return torch.where(
            observations < low,
            below,
            torch.where(observations > high, above, inside),
        )
    
    def _compute_crps_decomposition(self, observations: Tensor, ensemble: Tensor) -> Dict[str, float]:
        """
        Compute CRPS reliability-resolution-uncertainty decomposition.
        
        Mathematical Expression:
            CRPS(F,y) = integral BS_t(F,y) dt
            BS_t = Reliability_t - Resolution_t + Uncertainty_t
            CRPS = Reliability - Resolution + Uncertainty

        The integral is exact for the empirical ensemble CDF because forecast
        probabilities and binary outcomes are constant between consecutive
        unique forecast and observation values.
            
        References:
            - Murphy, A.H. (1973). A new vector partition of the probability
              score. Journal of Applied Meteorology 12, 595-600.
            - Hersbach, H. (2000). Decomposition of the continuous ranked probability
              score for ensemble forecasts. Weather Forecast. 15, 559-570.
        """
        breakpoints = torch.unique(
            torch.cat((observations, ensemble.reshape(-1))),
            sorted=True,
        )
        reliability = 0.0
        resolution = 0.0
        uncertainty = 0.0

        for left, right in zip(breakpoints[:-1], breakpoints[1:]):
            width = float(right - left)
            threshold = (left + right) / 2
            probabilities = (ensemble <= threshold).float().mean(dim=1)
            outcomes = (observations <= threshold).float()
            base_rate = float(outcomes.mean())

            threshold_reliability = 0.0
            threshold_resolution = 0.0
            for probability in torch.unique(probabilities):
                mask = probabilities == probability
                group_weight = float(mask.float().mean())
                group_forecast = float(probabilities[mask].mean())
                group_outcome = float(outcomes[mask].mean())
                threshold_reliability += (
                    group_weight * (group_forecast - group_outcome) ** 2
                )
                threshold_resolution += (
                    group_weight * (group_outcome - base_rate) ** 2
                )

            reliability += width * threshold_reliability
            resolution += width * threshold_resolution
            uncertainty += width * base_rate * (1.0 - base_rate)

        return {
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty,
        }
    
    def diagnostic_plots(self,
                        observations: Tensor,
                        predictions: Union[Tensor, Dict[str, Tensor]],
                        save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate CRPS diagnostic plots.
        
        Parameters:
            observations (Tensor): Observed values
            predictions: Forecast ensemble or parameters
            save_path (Optional[str]): Path to save plots
            
        Returns:
            Dict[str, Any]: Plot data and statistics
        """
        # Compute CRPS values
        crps_result = self.compute_score(observations, predictions)
        crps_values = crps_result['crps_values']
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('CRPS Diagnostic Plots', fontsize=16)
        
        # 1. CRPS distribution
        crps_numpy = crps_values.detach().cpu().numpy()
        axes[0, 0].hist(crps_numpy, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 0].axvline(crps_numpy.mean(), color='red', linestyle='--', 
                          label=f'Mean CRPS = {crps_numpy.mean():.3f}')
        axes[0, 0].set_xlabel('CRPS Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('CRPS Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. CRPS vs observation value
        obs_numpy = observations.detach().cpu().numpy()
        axes[0, 1].scatter(obs_numpy, crps_numpy, alpha=0.6, s=20, color='blue')
        axes[0, 1].set_xlabel('Observation Value')
        axes[0, 1].set_ylabel('CRPS Value')
        axes[0, 1].set_title('CRPS vs Observation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. CRPS time series
        axes[1, 0].plot(crps_numpy, 'b-', alpha=0.7, linewidth=1)
        axes[1, 0].axhline(crps_numpy.mean(), color='red', linestyle='--', label='Mean CRPS')
        axes[1, 0].set_xlabel('Forecast Index')
        axes[1, 0].set_ylabel('CRPS Value')
        axes[1, 0].set_title('CRPS Time Series')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. CRPS skill vs climatology
        skill_score = None
        if isinstance(predictions, Tensor) and predictions.shape[1] > 1:
            # Compute climatological CRPS (using observation climatology)
            climatology = observations.unsqueeze(0).expand(
                len(observations),
                -1,
            )
            clim_crps = self._compute_ensemble_crps(observations, climatology)
            valid_reference = clim_crps > 0
            safe_reference = torch.where(
                valid_reference,
                clim_crps,
                torch.ones_like(clim_crps),
            )
            skill_score = torch.where(
                valid_reference,
                1 - crps_values / safe_reference,
                torch.full_like(crps_values, torch.nan),
            )

            skill_numpy = skill_score.detach().cpu().numpy()
            finite_skill = skill_numpy[np.isfinite(skill_numpy)]
            if finite_skill.size > 0:
                axes[1, 1].hist(finite_skill, bins=30, alpha=0.7, color='green', edgecolor='black')
                axes[1, 1].axvline(finite_skill.mean(), color='darkgreen', linestyle='--',
                                  label=f'Mean Skill = {finite_skill.mean():.3f}')
                axes[1, 1].axvline(0, color='black', linestyle='-', alpha=0.5, label='No skill')
                axes[1, 1].set_xlabel('CRPS Skill Score')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('CRPS Skill Score vs Climatology')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(
                    0.5,
                    0.5,
                    'Skill score undefined\nfor zero-CRPS climatology',
                    ha='center',
                    va='center',
                    transform=axes[1, 1].transAxes,
                )
                axes[1, 1].set_title('CRPS Skill Score')
        else:
            axes[1, 1].text(0.5, 0.5, 'Skill score\nrequires ensemble\nforecasts', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('CRPS Skill Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"CRPS plots saved to {save_path}")
        
        return {
            'crps_values': crps_values,
            'crps_mean': crps_result['crps_mean'],
            'crps_std': crps_result['crps_std'],
            'skill_score': skill_score,
            'figure': fig
        }


def pit_test(observations: Tensor,
             forecast_ensemble: Tensor,
             alpha: Optional[float] = None) -> StochasticValidationResult:
    """
    Convenience function for PIT consistency test.
    
    Parameters:
        observations (Tensor): Observed values [n_obs]
        forecast_ensemble (Tensor): Ensemble forecasts [n_obs, n_members]
        alpha (Optional[float]): Significance level for tests; None reads
            validation.stochastic_scores.pit_significance_level
        
    Returns:
        StochasticValidationResult: Test results and diagnostics
        
    Example:
        >>> obs = torch.randn(100)
        >>> forecasts = obs.unsqueeze(1) + 0.1 * torch.randn(100, 20)  
        >>> result = pit_test(obs, forecasts)
        >>> print(f"KS p-value: {result.metadata['ks_pvalue']:.3f}")
    """
    validator = PITValidator(alpha=alpha)
    scores = validator.compute_score(observations, forecast_ensemble)
    
    # Determine overall significance
    primary_pvalue = scores['ks_pvalue']
    
    return StochasticValidationResult(
        score=scores['pit_mean'],
        decomposition={
            'pit_std': scores['pit_std'],
            'deviation_from_uniform': abs(scores['pit_mean'] - 0.5)
        },
        pvalue=primary_pvalue,
        sample_size=len(observations),
        metadata=scores
    )


def crps_score(observations: Tensor,
               predictions: Union[Tensor, Dict[str, Tensor]],
               fair: bool = False) -> StochasticValidationResult:
    """
    Convenience function for CRPS computation.
    
    Parameters:
        observations (Tensor): Observed values [n_obs]
        predictions (Union[Tensor, Dict]): Ensemble forecasts or distribution params
        fair (bool): Use bias-corrected fair CRPS
        
    Returns:
        StochasticValidationResult: CRPS scores and decomposition
        
    Example:
        >>> obs = torch.randn(100)
        >>> ensemble = obs.unsqueeze(1) + 0.2 * torch.randn(100, 50)
        >>> result = crps_score(obs, ensemble) 
        >>> print(f"Mean CRPS: {result.score:.4f}")
    """
    validator = CRPSValidator(decompose=True)
    scores = validator.compute_score(observations, predictions, fair_crps=fair)
    
    return StochasticValidationResult(
        score=scores['crps_mean'],
        decomposition={
            key: scores[key]
            for key in (
                'reliability',
                'resolution',
                'uncertainty',
            )
            if key in scores
        },
        sample_size=len(observations),
        metadata=scores
    )


def coverage_probability(observations: Tensor,
                        lower_bounds: Tensor, 
                        upper_bounds: Tensor,
                        nominal_coverage: Optional[float] = None) -> StochasticValidationResult:
    """
    Compute empirical coverage probability for prediction intervals.
    
    Parameters:
        observations (Tensor): Observed values [n_obs]
        lower_bounds (Tensor): Lower interval bounds [n_obs]
        upper_bounds (Tensor): Upper interval bounds [n_obs]
        nominal_coverage (float, optional): Expected coverage probability;
            None reads validation.stochastic_scores.nominal_interval_coverage
        
    Returns:
        StochasticValidationResult: Coverage statistics and significance test
        
    Mathematical Expression:
        Coverage = (1/n) Sigma? I(lower_i <= y_i <= upper_i)
        where I is indicator function
        
    References:
        - Wilks, D.S. (2011). Statistical Methods in the Atmospheric Sciences.
    """
    _validate_finite_1d("Observations", observations)
    _validate_finite_1d("lower_bounds", lower_bounds)
    _validate_finite_1d("upper_bounds", upper_bounds)
    _validate_same_shape("Observations", observations, "lower_bounds", lower_bounds)
    _validate_same_shape("Observations", observations, "upper_bounds", upper_bounds)
    if nominal_coverage is None:
        # The nominal coverage of the central interval is a declared YAML value
        # (Gneiting and Raftery 2007); there is no code default.
        nominal_coverage = float(get_global_manager().get(
            "validation.stochastic_scores.nominal_interval_coverage", required=True
        ))
    if not math.isfinite(nominal_coverage) or not 0.0 < nominal_coverage < 1.0:
        raise ValueError(
            "nominal_coverage must be finite and strictly between 0 and 1"
        )
    if bool((lower_bounds > upper_bounds).any()):
        raise ValueError("Every lower bound must be less than or equal to its upper bound")

    # Check which observations fall within bounds
    within_bounds = (observations >= lower_bounds) & (observations <= upper_bounds)
    empirical_coverage = within_bounds.float().mean().item()
    
    # Binomial test for coverage probability
    n_obs = len(observations)
    n_covered = int(within_bounds.sum().item())

    # Two-sided binomial test
    pvalue = float(
        stats.binomtest(
            n_covered,
            n_obs,
            nominal_coverage,
            alternative='two-sided',
        ).pvalue
    )
    
    # References: Clopper, C.J. & Pearson, E.S. (1934). Biometrika 26, 404-413
    #   (exact binomial confidence interval).
    # Mathematical expression: [B(alpha/2; k, n-k+1), B(1-alpha/2; k+1, n-k)],
    #   alpha = 1 - c, lower limit 0 at k = 0 and upper limit 1 at k = n.
    # Parameter meanings: k covered observations out of n; B the beta quantile;
    #   c = validation.stochastic_scores.coverage_interval_confidence_level.
    # Units: probabilities (dimensionless).
    # Domain/constraints: 0 < c < 1 (YAML bounds [0.5, 0.9999]); n >= 1.
    # Sign convention: lower <= k/n <= upper.
    confidence_level = float(get_global_manager().get(
        "validation.stochastic_scores.coverage_interval_confidence_level", required=True
    ))
    alpha = 1.0 - confidence_level
    ci_lower = (
        0.0
        if n_covered == 0
        else float(stats.beta.ppf(alpha / 2, n_covered, n_obs - n_covered + 1))
    )
    ci_upper = (
        1.0
        if n_covered == n_obs
        else float(
            stats.beta.ppf(
                1 - alpha / 2,
                n_covered + 1,
                n_obs - n_covered,
            )
        )
    )

    return StochasticValidationResult(
        score=empirical_coverage,
        decomposition={
            'nominal_coverage': nominal_coverage,
            'coverage_deviation': abs(empirical_coverage - nominal_coverage),
            'n_covered': n_covered,
            'interval_width_mean': float((upper_bounds - lower_bounds).mean())
        },
        pvalue=pvalue,
        confidence_interval=(ci_lower, ci_upper),
        sample_size=n_obs,
        metadata={
            'within_bounds': within_bounds,
            'coverage_ratio': empirical_coverage / nominal_coverage
        }
    )


def brier_score(observations: Tensor,
                forecast_probabilities: Tensor,
                threshold: float) -> StochasticValidationResult:
    """
    Compute Brier score for probability forecasts of binary events.
    
    Parameters:
        observations (Tensor): Observed values [n_obs]
        forecast_probabilities (Tensor): Forecast probabilities [n_obs]
        threshold (float): Threshold for binary event definition
        
    Returns:
        StochasticValidationResult: Brier score and reliability decomposition
        
    Mathematical Expression:
        BS = (1/n) Sigma? (p? - o?)^2
        where p? is forecast probability, o? in {0,1} is binary outcome
        
    Decomposition:
        BS = Reliability - Resolution + Uncertainty
        
    References:
        - Brier, G.W. (1950). Verification of forecasts expressed in terms of probability.
          Mon. Wea. Rev. 78, 1-3.
    """
    _validate_finite_1d("Observations", observations)
    _validate_finite_1d("forecast_probabilities", forecast_probabilities)
    _validate_same_shape(
        "Observations",
        observations,
        "forecast_probabilities",
        forecast_probabilities,
    )
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if bool(
        ((forecast_probabilities < 0) | (forecast_probabilities > 1)).any()
    ):
        raise ValueError("forecast_probabilities must lie in [0, 1]")

    # Convert observations to binary outcomes
    binary_outcomes = (observations > threshold).float()
    
    # Compute Brier score
    bs_values = (forecast_probabilities - binary_outcomes) ** 2
    bs_mean = bs_values.mean().item()
    
    # Murphy decomposition (simplified)
    base_rate = binary_outcomes.mean().item()
    
    # Murphy decomposition over exact forecast-probability groups.
    reliability = 0.0
    resolution = 0.0
    for probability in torch.unique(forecast_probabilities):
        mask = forecast_probabilities == probability
        group_weight = mask.float().mean().item()
        group_forecast = forecast_probabilities[mask].mean().item()
        group_outcome = binary_outcomes[mask].mean().item()
        reliability += group_weight * (group_forecast - group_outcome) ** 2
        resolution += group_weight * (group_outcome - base_rate) ** 2

    uncertainty = base_rate * (1 - base_rate)
    
    return StochasticValidationResult(
        score=bs_mean,
        decomposition={
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty,
            'base_rate': base_rate
        },
        sample_size=len(observations),
        metadata={
            'bs_values': bs_values,
            'binary_outcomes': binary_outcomes,
            'threshold': threshold
        }
    )


def quantile_score(observations: Tensor,
                  forecast_quantiles: Tensor,
                  quantile_levels: Tensor) -> StochasticValidationResult:
    """
    Compute quantile score (pinball loss) for quantile forecasts.
    
    Parameters:
        observations (Tensor): Observed values [n_obs]
        forecast_quantiles (Tensor): Forecast quantiles [n_obs, n_quantiles] 
        quantile_levels (Tensor): Quantile levels [n_quantiles]
        
    Returns:
        StochasticValidationResult: Quantile scores by level
        
    Mathematical Expression:
        QS(tau) = (1/n) Sigma? rho_tau(y? - q_tau,i)
        where rho_tau(u) = u(tau - I(u<0)) is quantile loss function
        
    References:
        - Koenker, R. (2005). Quantile Regression. Cambridge University Press.
        - Gneiting, T. & Raftery, A.E. (2007). Strictly proper scoring rules.
    """
    _validate_finite_1d("Observations", observations)
    if not isinstance(forecast_quantiles, Tensor):
        raise TypeError("forecast_quantiles must be a torch.Tensor")
    if forecast_quantiles.dim() != 2:
        raise ValueError(
            "forecast_quantiles must be 2D [n_obs, n_quantiles], "
            f"got {forecast_quantiles.shape}"
        )
    if forecast_quantiles.numel() == 0:
        raise ValueError("forecast_quantiles must be non-empty")
    if not bool(torch.isfinite(forecast_quantiles).all()):
        raise ValueError("forecast_quantiles must contain only finite values")
    _validate_finite_1d("quantile_levels", quantile_levels)

    n_obs, n_quantiles = forecast_quantiles.shape
    if n_obs != len(observations):
        raise ValueError(
            "Observations and forecast_quantiles must have the same number "
            f"of samples, got {len(observations)} and {n_obs}"
        )
    if len(quantile_levels) != n_quantiles:
        raise ValueError(f"Quantile levels size mismatch: {len(quantile_levels)} vs {n_quantiles}")
    if bool(((quantile_levels <= 0) | (quantile_levels >= 1)).any()):
        raise ValueError("quantile_levels must be strictly between 0 and 1")
    if n_quantiles > 1 and bool(
        (quantile_levels[1:] <= quantile_levels[:-1]).any()
    ):
        raise ValueError("quantile_levels must be strictly increasing")
    if n_quantiles > 1 and bool(
        (forecast_quantiles[:, 1:] < forecast_quantiles[:, :-1]).any()
    ):
        raise ValueError("forecast_quantiles must be nondecreasing across levels")
    
    # Compute quantile scores for each level
    errors = observations.unsqueeze(1) - forecast_quantiles  # [n_obs, n_quantiles]
    
    # Quantile loss: tau * error if error > 0, (tau-1) * error if error <= 0
    positive_errors = (errors > 0).float()
    quantile_losses = quantile_levels.unsqueeze(0) * errors * positive_errors + \
                     (quantile_levels.unsqueeze(0) - 1) * errors * (1 - positive_errors)
    
    # Mean score for each quantile level
    mean_scores = quantile_losses.mean(dim=0)  # [n_quantiles]
    
    # Overall score (integral over all quantiles)
    if n_quantiles > 1:
        # Approximate integral using trapezoidal rule
        dq = quantile_levels[1:] - quantile_levels[:-1]
        integral_score = torch.sum(0.5 * (mean_scores[:-1] + mean_scores[1:]) * dq).item()
    else:
        integral_score = mean_scores[0].item()
    aggregation_method = (
        "trapezoidal_pinball_integral"
        if n_quantiles > 1
        else "single_level_pinball_loss"
    )
    
    return StochasticValidationResult(
        score=integral_score,
        decomposition={
            quantile_label(q): float(score)
            for q, score in zip(quantile_levels, mean_scores)
        },
        sample_size=n_obs,
        metadata={
            'quantile_levels': quantile_levels,
            'mean_scores_by_level': mean_scores,
            'forecast_quantiles': forecast_quantiles,
            'aggregation_method': aggregation_method,
            'quantile_level_range': (
                float(quantile_levels[0]),
                float(quantile_levels[-1]),
            ),
            'is_full_crps': False,
            'score_interpretation': (
                "Unnormalized pinball-loss integral over supplied "
                "quantile levels"
            ),
        }
    )

