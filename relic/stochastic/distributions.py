"""
Parameter Distributions and Spatial Correlation for RELIC

This module implements parameter distributions with optional spatial correlation
for uncertainty quantification and Monte Carlo simulations. It integrates with
the unified parameter management system and supports CF-1.8 compliant output.

Key Features:
    - Multiple distribution types with scientific bounds
    - Spatial correlation using exponential kernels  
    - Integration with parameters/manager.py for ParameterSpec
    - Memory-efficient sampling for large spatial domains
    - CF-1.8 compliant variable naming and units
    - Automatic validation and constraint checking

Mathematical Framework:
    Parameter distribution: theta ~ D(mu, sigma, bounds) where D is distribution family
    Spatial correlation: C(h) = exp(-h/L) where h is distance, L is correlation length
    Correlated sampling: theta(x) = mu + sigma * Sigma_i sqrt(lambda_i) * phi_i(x) * xi_i
    where lambda_i, phi_i are eigenvalues/eigenfunctions of correlation operator

References:
    - Rasmussen, C.E. & Williams, C.K.I. (2006). Gaussian Processes for Machine Learning
    - Saltelli, A. et al. (2008). Global Sensitivity Analysis: The Primer
    - Soize, C. (2017). Uncertainty Quantification. Springer
    - Rosolem, R. et al. (2012). A fully multiple-criteria implementation
      of the Sobol method. J. Geophys. Res. 117, D07103.

Authors: Yuchong Long  
"""

from typing import Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import math

import torch
from torch import Tensor
import torch.distributions as dist
import numpy as np

from .category import ParameterUncertainty, ProcessMetadata
from ..parameters import ParameterSpec

logger = logging.getLogger(__name__)


class DistributionType(Enum):
    """Supported parameter distribution types."""
    UNIFORM = "uniform"
    NORMAL = "normal" 
    LOGNORMAL = "lognormal"
    BETA = "beta"
    GAMMA = "gamma"
    TRIANGULAR = "triangular"
    TRUNCATED_NORMAL = "truncated_normal"


class CorrelationKernel(Enum):
    """Spatial correlation kernel functions."""
    EXPONENTIAL = "exponential"      # exp(-h/L)
    SQUARED_EXPONENTIAL = "squared_exponential"  # exp(-h^2/L^2) 
    MATERN_32 = "matern_3_2"        # (1 + sqrt(3)h/L) * exp(-sqrt(3)h/L)
    MATERN_52 = "matern_5_2"        # (1 + sqrt(5)h/L + 5h^2/3L^2) * exp(-sqrt(5)h/L)


@dataclass
class DistributionSpec:
    """
    Specification for parameter distribution.
    
    Parameters:
        dist_type (DistributionType): Distribution family
        params (Dict[str, float]): Distribution parameters (mean, std, etc.)
        bounds (Tuple[float, float]): Hard bounds for parameter values
        log_scale (bool): Sample in log space then transform
        integer_valued (bool): Round samples to integers
        correlation_length (Optional[float]): Spatial correlation length [same units as coordinates];
            None declares an uncorrelated parameter
        kernel (Optional[CorrelationKernel]): Spatial correlation kernel
            function. Required with a correlation_length; None (and only
            None) without one. No kernel is chosen for the caller.
        correlation_nugget (Optional[float]): Declared nugget tau added to
            the correlation matrix, K + tau I (Cressie 1993, nugget effect;
            Rasmussen and Williams 2006, jitter for numerically singular
            kernels). Dimensionless, finite, >= 0, where 0 declares an exact
            kernel. Required with a correlation_length; None (and only None)
            without one. No regularization is added unless declared here.
        
    Mathematical Expression:
        For normal distribution: theta ~ N(mu, sigma^2) subject to theta in [a, b]
        For lognormal: log(theta) ~ N(mu, sigma^2)
        For uniform: theta ~ U(a, b)
        For beta: theta ~ Beta(alpha, beta) scaled to [a, b]
        
    Units:
        All parameters maintain CF-1.8 units from ParameterSpec
    """
    dist_type: DistributionType
    params: Dict[str, float]
    bounds: Tuple[float, float] = (-float('inf'), float('inf'))
    log_scale: bool = False
    integer_valued: bool = False 
    correlation_length: Optional[float] = None
    kernel: Optional[CorrelationKernel] = None
    correlation_nugget: Optional[float] = None

    def __post_init__(self) -> None:
        """A correlated field declares kernel and nugget; an uncorrelated one neither."""
        if self.correlation_length is None:
            declared = [
                name for name in ("kernel", "correlation_nugget")
                if getattr(self, name) is not None
            ]
            if declared:
                raise ValueError(
                    f"{', '.join(declared)} describe a spatial random field and "
                    "need a correlation_length; an uncorrelated parameter "
                    "declares neither"
                )
            return
        if self.kernel is None:
            raise ValueError(
                "A correlated parameter (correlation_length set) must declare "
                "its correlation kernel; no kernel is chosen for it"
            )
        if not isinstance(self.kernel, CorrelationKernel):
            raise TypeError(
                f"kernel must be a CorrelationKernel, got {type(self.kernel).__name__}"
            )
        if self.correlation_nugget is None:
            raise ValueError(
                "A correlated parameter (correlation_length set) must declare "
                "correlation_nugget (0 declares an exact kernel)"
            )
        nugget = self.correlation_nugget
        if (
            isinstance(nugget, bool)
            or not isinstance(nugget, (int, float))
            or not math.isfinite(float(nugget))
            or float(nugget) < 0.0
        ):
            raise ValueError(
                f"correlation_nugget must be a finite number >= 0, got {nugget!r}"
            )


class ParameterDistribution(ParameterUncertainty):
    """
    Parameter distribution class with spatial correlation support.
    
    Implements various probability distributions for parameters with optional
    spatial correlation structure. Designed to work with ParameterSpec from
    the unified parameter manager.
    
    Mathematical Expression:
        Base distribution: theta ~ p(theta | distribution_params)
        Spatial process: theta(x) ~ GP(mu(x), K(x, x'))
        where K(x, x') is correlation kernel function
        
    Units:
        Maintains CF-1.8 unit compliance throughout sampling
        
    References:
        - Santner, T.J. et al. (2003). Design and Analysis of Computer Experiments
        - O'Hagan, A. & Forster, J. (2004). Kendall's Advanced Theory of Statistics
    """
    
    def __init__(self, 
                 parameter_specs: Dict[str, Union[ParameterSpec, DistributionSpec]],
                 spatial_coords: Optional[Tensor] = None,
                 device: torch.device = torch.device("cpu")):
        """
        Initialize parameter distribution.
        
        Parameters:
            parameter_specs (Dict): Parameter specifications and distributions
            spatial_coords (Optional[Tensor]): Spatial coordinates [n_points, n_dims] 
            device (torch.device): Computation device
            
        Mathematical Expression:
            Parameter space: Theta = {theta_1, theta_2, ..., theta?}
            Spatial domain: X subset ?? where d in {1,2,3}
            Joint distribution: p(Theta|X) = product? p(theta?|x?)
        """
        # Initialize metadata
        metadata = ProcessMetadata(
            name="ParameterDistribution",
            description="Parameter uncertainty with spatial correlation",
            units={"parameters": "various", "samples": "dimensionless"},
            parameters={"n_params": len(parameter_specs)},
            references="Saltelli et al. (2008), Rasmussen & Williams (2006)"
        )
        super().__init__(metadata)
        
        self.parameter_specs = parameter_specs
        self.spatial_coords = spatial_coords
        self.device = device
        
        # Process parameter specifications
        self.param_keys = list(parameter_specs.keys())
        self.distributions = {}
        self.correlation_matrices = {}
        
        self._setup_distributions()
        
        if spatial_coords is not None:
            self._precompute_correlation_matrices()
    
    def _setup_distributions(self):
        """Set up PyTorch distributions for each parameter."""
        for key, spec in self.parameter_specs.items():
            if isinstance(spec, ParameterSpec):
                # A ParameterSpec declares only bounds. Finite bounds give the
                # uniform (maximum-entropy) distribution over them, the usual
                # choice of global sensitivity analysis when only the range is
                # known (Saltelli et al. 2008). Without two finite bounds no
                # spread is invented: a DistributionSpec is required.
                bounds = spec.bounds
                if not (math.isfinite(float(bounds[0])) and math.isfinite(float(bounds[1]))):
                    raise ValueError(
                        f"ParameterSpec '{key}' has bounds {tuple(bounds)}; an "
                        "uncertainty distribution needs two finite bounds or an "
                        "explicit DistributionSpec"
                    )
                else:
                    dist_spec = DistributionSpec(
                        dist_type=DistributionType.UNIFORM,
                        params={'low': bounds[0], 'high': bounds[1]},
                        bounds=bounds
                    )
            else:
                dist_spec = spec
            
            # Create PyTorch distribution
            self.distributions[key] = self._create_torch_distribution(dist_spec)
    
    def _create_torch_distribution(self, spec: DistributionSpec) -> dist.Distribution:
        """Create PyTorch distribution from specification."""
        if spec.dist_type == DistributionType.UNIFORM:
            return dist.Uniform(
                low=torch.tensor(spec.params['low'], device=self.device),
                high=torch.tensor(spec.params['high'], device=self.device)
            )
        elif spec.dist_type == DistributionType.NORMAL:
            return dist.Normal(
                loc=torch.tensor(spec.params['loc'], device=self.device),
                scale=torch.tensor(spec.params['scale'], device=self.device)
            )
        elif spec.dist_type == DistributionType.LOGNORMAL:
            return dist.LogNormal(
                loc=torch.tensor(spec.params['loc'], device=self.device),
                scale=torch.tensor(spec.params['scale'], device=self.device)
            )
        elif spec.dist_type == DistributionType.BETA:
            alpha = torch.tensor(spec.params['alpha'], device=self.device)
            beta = torch.tensor(spec.params['beta'], device=self.device)
            low = spec.params.get('low', 0.0)
            high = spec.params.get('high', 1.0)
            
            # Scale beta distribution to [low, high]
            base_beta = dist.Beta(alpha, beta)
            return dist.TransformedDistribution(
                base_beta,
                dist.transforms.AffineTransform(loc=low, scale=high-low)
            )
        elif spec.dist_type == DistributionType.GAMMA:
            return dist.Gamma(
                concentration=torch.tensor(spec.params['concentration'], device=self.device),
                rate=torch.tensor(spec.params['rate'], device=self.device)
            )
        elif spec.dist_type in (
            DistributionType.TRIANGULAR,
            DistributionType.TRUNCATED_NORMAL,
        ):
            # No torch distribution implements these families here. The
            # former code returned Uniform(low, high) for 'triangular' and a
            # logit transform of a normal for 'truncated_normal'; neither is
            # substituted any more.
            raise NotImplementedError(
                f"Distribution type '{spec.dist_type.value}' is not implemented "
                "by ParameterDistribution"
            )
        else:
            raise ValueError(f"Unsupported distribution type: {spec.dist_type}")
    
    def _precompute_correlation_matrices(self):
        """Precompute spatial correlation matrices for correlated parameters."""
        if self.spatial_coords is None:
            return
        
        for key, spec in self.parameter_specs.items():
            if isinstance(spec, DistributionSpec) and spec.correlation_length is not None:
                nugget = float(spec.correlation_nugget)
                if not math.isfinite(nugget) or nugget < 0.0:
                    raise ValueError(
                        f"correlation_nugget of '{key}' must be finite and "
                        f">= 0, got {spec.correlation_nugget!r}"
                    )
                corr_matrix = self._compute_correlation_matrix(
                    self.spatial_coords, 
                    spec.correlation_length,
                    spec.kernel
                )
                
                # Store correlation info
                self.correlation_matrices[key] = {
                    'matrix': corr_matrix,
                    'length': spec.correlation_length,
                    'kernel': spec.kernel,
                    'nugget': nugget,
                }
                
                logger.debug(f"Precomputed correlation matrix for {key}: {corr_matrix.shape}")
    
    def _compute_correlation_matrix(self, 
                                   coords: Tensor,
                                   length_scale: float,
                                   kernel: CorrelationKernel) -> Tensor:
        """
        Compute spatial correlation matrix using specified kernel.
        
        Parameters:
            coords (Tensor): Coordinates [n_points, n_dims]
            length_scale (float): Correlation length scale
            kernel (CorrelationKernel): Kernel function type
            
        Returns:
            Tensor: Correlation matrix [n_points, n_points]
            
        Mathematical Expression:
            For exponential kernel: K(x, x') = exp(-||x - x'|| / L)
            For squared exponential: K(x, x') = exp(-||x - x'||^2 / L^2)
            For Matern-3/2: K(x, x') = (1 + sqrt3r/L) exp(-sqrt3r/L)
            where r = ||x - x'||
        """
        # Compute pairwise distances
        coords_expanded_i = coords.unsqueeze(1)  # [n, 1, d]
        coords_expanded_j = coords.unsqueeze(0)  # [1, n, d]
        distances = torch.norm(coords_expanded_i - coords_expanded_j, dim=2)  # [n, n]
        
        return self._kernel_values(distances, length_scale, kernel)

    @staticmethod
    def _kernel_values(distances: Tensor,
                       length_scale: float,
                       kernel: CorrelationKernel) -> Tensor:
        """Evaluate the correlation kernel on a tensor of distances.

        Single kernel implementation shared by the full correlation matrix
        and the Nystroem cross-correlation (formulas in
        _compute_correlation_matrix).

        Raises:
            ValueError: for a kernel outside CorrelationKernel.
        """
        if kernel == CorrelationKernel.EXPONENTIAL:
            return torch.exp(-distances / length_scale)
        if kernel == CorrelationKernel.SQUARED_EXPONENTIAL:
            return torch.exp(-distances**2 / (length_scale**2))
        if kernel == CorrelationKernel.MATERN_32:
            scaled_dist = distances * np.sqrt(3) / length_scale
            return (1 + scaled_dist) * torch.exp(-scaled_dist)
        if kernel == CorrelationKernel.MATERN_52:
            scaled_dist = distances * np.sqrt(5) / length_scale
            return (1 + scaled_dist + scaled_dist**2 / 3) * torch.exp(-scaled_dist)
        raise ValueError(f"Unsupported kernel: {kernel}")
    
    def sample(self,
               input_state: Tensor,
               num_samples: int = 1,
               seed: Optional[int] = None,
               **kwargs) -> Dict[str, Tensor]:
        """
        Generate parameter samples with optional spatial correlation.
        
        Parameters:
            input_state (Tensor): Not used for parameter distributions  
            num_samples (int): Number of sample sets to generate
            seed (Optional[int]): Random seed for reproducibility
            **kwargs: Additional sampling options
            
        Returns:
            Dict[str, Tensor]: Parameter samples
                Format: {param_name: tensor[num_samples, n_spatial_points]}
                
        Mathematical Expression:
            Independent: theta?? ~ p(theta?) for i=1,...,n_params, j=1,...,num_samples
            Correlated: theta?(x) ~ GP(mu?, sigma?^2 K(x,x')) 
            where K is spatial correlation matrix
            
        Units:
            Output tensors maintain CF-1.8 units from ParameterSpec metadata
        """
        if seed is None:
            return self._draw_samples(num_samples)
        # A seeded draw runs on a forked generator state, so the caller's
        # global Torch RNG streams are left unchanged. Only the forked
        # generators are seeded: torch.manual_seed would also reseed every
        # CUDA generator outside the fork.
        fork_devices = [] if self.device.type == "cpu" else [self.device]
        with torch.random.fork_rng(devices=fork_devices):
            torch.default_generator.manual_seed(seed)
            if self.device.type == "cuda":
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
            return self._draw_samples(num_samples)

    def _draw_samples(self, num_samples: int) -> Dict[str, Tensor]:
        """Draw one set of samples from the current generator state."""
        samples = {}
        n_spatial = self.spatial_coords.shape[0] if self.spatial_coords is not None else 1
        
        for key in self.param_keys:
            distribution = self.distributions[key]
            
            if key in self.correlation_matrices:
                # Spatially correlated sampling
                samples[key] = self._sample_correlated(
                    key, distribution, num_samples, n_spatial
                )
            else:
                # Independent sampling
                sample_shape = (num_samples, n_spatial)
                samples[key] = distribution.sample(sample_shape)
            
            # Post-processing in value space: a log-scale sample is first
            # transformed to the parameter value, then the value bounds are
            # applied, then integer rounding (the former order clamped and
            # rounded the log value and exponentiated afterwards).
            if isinstance(self.parameter_specs[key], DistributionSpec):
                spec = self.parameter_specs[key]
                
                if spec.log_scale:
                    samples[key] = torch.exp(samples[key])

                if spec.bounds[0] != -float('inf') or spec.bounds[1] != float('inf'):
                    samples[key] = torch.clamp(
                        samples[key],
                        min=spec.bounds[0],
                        max=spec.bounds[1]
                    )
                
                if spec.integer_valued:
                    samples[key] = torch.round(samples[key])
        
        return samples
    
    def _sample_correlated(self,
                          param_key: str,
                          distribution: dist.Distribution,
                          num_samples: int,
                          n_spatial: int) -> Tensor:
        """
        Generate spatially correlated samples using Cholesky decomposition.
        
        Parameters:
            param_key (str): Parameter identifier
            distribution (dist.Distribution): Marginal distribution
            num_samples (int): Number of sample realizations
            n_spatial (int): Number of spatial points
            
        Returns:
            Tensor: Correlated samples [num_samples, n_spatial]
            
        Mathematical Expression:
            Y ~ N(0, I), Theta = mu + sigma L Y
            where L L^T = K is Cholesky decomposition of correlation matrix
            Then Theta ~ N(mu, sigma^2 K) has desired correlation structure
        """
        corr_info = self.correlation_matrices[param_key]
        corr_matrix = corr_info['matrix']
        nugget = corr_info['nugget']

        # References:
        #   Rasmussen and Williams (2006), Gaussian Processes for Machine
        #   Learning, appendix A.2 (sampling by Cholesky factor, jitter);
        #   Cressie (1993), Statistics for Spatial Data (nugget effect).
        # Mathematical expression:
        #   L L^T = K + tau I, z ~ N(0, I), y = L z (y ~ N(0, K + tau I))
        # Parameter meanings:
        #   K: kernel correlation matrix of the spatial points; tau: the
        #   declared DistributionSpec.correlation_nugget (0 when none); L:
        #   lower Cholesky factor; z: independent standard normal draws.
        # Units:
        #   K, tau, z, y dimensionless (unit-variance field).
        # Domain/constraints:
        #   K + tau I must be positive definite. A failed factorization is
        #   an error reporting the minimum eigenvalue; it is never replaced
        #   by an eigenvalue-clamped projection or an undeclared jitter.
        # Sign convention:
        #   None; symmetric correlation structure.
        reg_matrix = corr_matrix
        if nugget > 0.0:
            reg_matrix = corr_matrix + nugget * torch.eye(
                n_spatial, device=corr_matrix.device, dtype=corr_matrix.dtype
            )
        try:
            L = torch.linalg.cholesky(reg_matrix)
        except torch.linalg.LinAlgError as err:
            min_eigenvalue = float(torch.linalg.eigvalsh(reg_matrix).min())
            raise ValueError(
                f"Correlation matrix of '{param_key}' is not positive definite: "
                f"minimum eigenvalue {min_eigenvalue:.3e} for {n_spatial} points "
                f"(kernel {corr_info['kernel'].value}, length "
                f"{corr_info['length']}, nugget {nugget}); remove coincident "
                "points or declare DistributionSpec.correlation_nugget"
            ) from err

        # Independent standard normal draws in the dtype of the factor (the
        # field follows the coordinate/kernel precision).
        standard_samples = torch.randn(
            num_samples, n_spatial, device=L.device, dtype=L.dtype
        )
        
        # Apply correlation structure
        correlated_samples = torch.matmul(standard_samples, L.T)
        
        # Transform to marginal distribution
        # Use inverse CDF transform for non-Gaussian distributions
        if isinstance(distribution, (dist.Normal, dist.LogNormal)):
            # References: Rasmussen and Williams (2006); log-normal random
            #   fields as exp of a Gaussian field (Cressie 1993, 3.2).
            # Mathematical expression: Normal: theta = mu + sigma y;
            #   LogNormal(mu, sigma): theta = exp(mu + sigma y).
            # Parameter meanings: mu, sigma: loc and scale of the (underlying)
            #   normal law; y: correlated unit-variance field.
            # Units: theta in the parameter unit; mu, sigma in the unit of
            #   theta (normal) or of log(theta) (log-normal); y dimensionless.
            # Domain/constraints: sigma > 0 (checked by torch.distributions).
            # Sign convention: None.
            loc = distribution.loc.item() if distribution.loc.numel() == 1 else distribution.loc
            scale = distribution.scale.item() if distribution.scale.numel() == 1 else distribution.scale
            final_samples = loc + scale * correlated_samples
            if isinstance(distribution, dist.LogNormal):
                final_samples = torch.exp(final_samples)
        else:
            # For non-Gaussian, use Gaussian copula approach
            # Transform to uniform via standard normal CDF
            uniform_samples = torch.distributions.Normal(0, 1).cdf(correlated_samples)
            
            # Transform uniform to target distribution via inverse CDF
            final_samples = distribution.icdf(uniform_samples)
        
        return final_samples
    
    def sample_parameters(self,
                         param_specs: Dict[str, Any],
                         num_samples: int,
                         spatial_correlation: Optional[Tensor] = None,
                         seed: Optional[int] = None) -> Dict[str, Tensor]:
        """
        Sample parameters from distributions (implements ParameterUncertainty interface).
        
        Parameters:
            param_specs (Dict[str, Any]): Parameter specifications (ignored, uses internal specs)
            num_samples (int): Number of parameter sets
            spatial_correlation (Optional[Tensor]): External correlation matrix (ignored)
            seed (Optional[int]): Random seed
            
        Returns:
            Dict[str, Tensor]: Parameter samples with CF-1.8 naming
        """
        # Delegate to main sample method
        return self.sample(
            input_state=torch.empty(0),  # Not used
            num_samples=num_samples,
            seed=seed
        )
    
    def log_probability(self,
                       input_state: Tensor,
                       output_state: Tensor,
                       **kwargs) -> Tensor:
        """
        Compute log probability of parameter values.
        
        Parameters:
            input_state (Tensor): Not used for parameter distributions
            output_state (Tensor): Parameter values to evaluate
            **kwargs: Additional evaluation options
            
        Returns:
            Tensor: Log probability densities
            
        Mathematical Expression:
            For independent: log p(Theta) = Sigma? log p(theta?)
            For correlated: log p(Theta) = -1/2[(theta-mu)^T K^{-1} (theta-mu)] + const
        """
        if output_state.dim() != 2:
            raise ValueError(f"Expected 2D output_state [n_samples, n_params], got {output_state.shape}")
        
        n_samples = output_state.shape[0]
        log_probs = torch.zeros(n_samples, device=self.device)
        
        for i, key in enumerate(self.param_keys):
            param_values = output_state[:, i]
            distribution = self.distributions[key]
            
            if key in self.correlation_matrices:
                # The multivariate density of a correlated field is not
                # implemented; the independent marginal density is not
                # returned in its place.
                raise NotImplementedError(
                    f"log_probability of the spatially correlated parameter "
                    f"'{key}' is not implemented"
                )
            log_probs += distribution.log_prob(param_values)
        
        return log_probs
    
    def moments(self,
                input_state: Tensor,
                **kwargs) -> Dict[str, Tensor]:
        """
        Compute analytical moments for parameter distributions.
        
        Parameters:
            input_state (Tensor): Not used for parameter distributions
            **kwargs: Additional moment computation options
            
        Returns:
            Dict[str, Tensor]: Mean and variance for each parameter
            
        Mathematical Expression:
            mu? = E[theta?], sigma?^2 = Var[theta?]
            For correlated case: Cov[theta?(x), theta?(x')] = sigma?^2 K(x,x')
        """
        moments = {}
        
        for key in self.param_keys:
            distribution = self.distributions[key]
            
            # Analytical moments only: a distribution without them raises; a
            # Monte Carlo estimate is never returned in their place.
            try:
                mean = distribution.mean
                var = distribution.variance
            except NotImplementedError as err:
                raise NotImplementedError(
                    f"Analytical moments are not available for '{key}' "
                    f"({type(distribution).__name__})"
                ) from err
            
            moments[key] = {
                'mean': mean,
                'variance': var,
                'std': torch.sqrt(var)
            }
        
        return moments


class SpatialCorrelation:
    """
    Utility class for spatial correlation computations.
    
    Provides efficient methods for computing and working with spatial
    correlation matrices, including approximations for large domains.
    
    Mathematical Framework:
        Correlation function: rho(h) where h is distance
        Covariance kernel: K(x, x') = sigma^2 rho(||x - x'||)
        Spectral representation: K = Sigma? lambda? phi? phi?^T
        
    References:
        - Cressie, N. (1993). Statistics for Spatial Data. Wiley
        - Stein, M.L. (1999). Interpolation of Spatial Data. Springer
    """
    
    @staticmethod
    def exponential_correlation(distances: Tensor, length_scale: float) -> Tensor:
        """
        Compute exponential correlation: rho(h) = exp(-h/L).
        
        Parameters:
            distances (Tensor): Pairwise distances
            length_scale (float): Correlation length L
            
        Returns:
            Tensor: Correlation values
            
        Mathematical Expression:
            rho(h) = exp(-h/L) where h >= 0, L > 0
            
        Units:
            length_scale units must match distance units
        """
        return torch.exp(-distances / length_scale)
    
    @staticmethod
    def squared_exponential_correlation(distances: Tensor, length_scale: float) -> Tensor:
        """
        Compute squared exponential correlation: rho(h) = exp(-h^2/L^2).
        
        Parameters:
            distances (Tensor): Pairwise distances  
            length_scale (float): Correlation length L
            
        Returns:
            Tensor: Correlation values
            
        Mathematical Expression:
            rho(h) = exp(-h^2/L^2) where h >= 0, L > 0
        """
        return torch.exp(-distances**2 / (length_scale**2))
    
    @staticmethod
    def matern_correlation(distances: Tensor, 
                          length_scale: float,
                          nu: float = 1.5) -> Tensor:
        """
        Compute Matern correlation function.
        
        Parameters:
            distances (Tensor): Pairwise distances
            length_scale (float): Correlation length L
            nu (float): Smoothness parameter (1.5 or 2.5 common)
            
        Returns:
            Tensor: Correlation values
            
        Mathematical Expression:
            rho(h) = (2^{1-nu}/Gamma(nu)) (sqrt{2nu}h/L)^nu K_nu(sqrt{2nu}h/L)
            where K_nu is modified Bessel function of second kind
        """
        if length_scale <= 0:
            raise ValueError(f"length_scale must be positive, got {length_scale}")
        if nu <= 0:
            raise ValueError(f"nu must be positive for Matern kernel, got {nu}")

        if nu == 0.5:
            # Matern-1/2 kernel (exponential kernel)
            scaled_dist = distances / length_scale
            return torch.exp(-scaled_dist)
        elif nu == 1.5:
            # Matern-3/2 kernel
            scaled_dist = distances * np.sqrt(3) / length_scale
            return (1 + scaled_dist) * torch.exp(-scaled_dist)
        elif nu == 2.5:
            # Matern-5/2 kernel  
            scaled_dist = distances * np.sqrt(5) / length_scale
            return (1 + scaled_dist + scaled_dist**2 / 3) * torch.exp(-scaled_dist)
        else:
            # Generic Matern kernel:
            # rho(h) = (2^(1-nu) / Gamma(nu)) * (sqrt(2nu)h/L)^nu * K_nu(sqrt(2nu)h/L)
            # where K_nu is the modified Bessel function of the second kind.
            try:
                from scipy import special as scipy_special
            except (ImportError, ModuleNotFoundError) as exc:
                raise RuntimeError(
                    "scipy is required for generic Matern kernel evaluation when nu is not 0.5/1.5/2.5."
                ) from exc

            # References:
            #   Rasmussen and Williams (2006), eq. 4.14; Stein (1999).
            # Mathematical expression:
            #   rho(h) = 2^(1-nu)/Gamma(nu) s^nu K_nu(s), s = sqrt(2 nu) h / L,
            #   evaluated as 2^(1-nu)/Gamma(nu) exp(nu ln s - s) kve(nu, s)
            #   with kve(nu, s) = K_nu(s) e^s; rho(0) = 1.
            # Parameter meanings:
            #   h: distance; L: length scale; nu: smoothness; K_nu: modified
            #   Bessel function of the second kind.
            # Units:
            #   h and L in the coordinate unit; s and rho dimensionless.
            # Domain/constraints:
            #   h finite and >= 0 (otherwise ValueError), L > 0, nu > 0. The
            #   scaled form has no overflow: rho -> 0 for large s without any
            #   NaN or infinity replacement.
            # Sign convention:
            #   None; rho in [0, 1] up to rounding.
            if not bool(torch.isfinite(distances).all()) or bool((distances < 0).any()):
                raise ValueError("Matern distances must be finite and non-negative")
            scaled_dist = distances * math.sqrt(2.0 * nu) / length_scale
            scaled_cpu = scaled_dist.detach().cpu().to(dtype=torch.float64).numpy()

            coeff = (2.0 ** (1.0 - nu)) / scipy_special.gamma(nu)
            at_origin = scaled_cpu == 0.0
            positive = np.where(at_origin, 1.0, scaled_cpu)
            corr_np = coeff * np.exp(nu * np.log(positive) - positive) * scipy_special.kve(nu, positive)
            corr_np = np.where(at_origin, 1.0, corr_np)

            return torch.as_tensor(corr_np, dtype=distances.dtype, device=distances.device)
    
    @staticmethod
    def approximate_correlation_matrix(coords: Tensor,
                                     length_scale: float,
                                     kernel: CorrelationKernel = CorrelationKernel.EXPONENTIAL,
                                     n_eigenfunctions: int = 50) -> Tuple[Tensor, Tensor]:
        """
        Approximate large correlation matrices using truncated eigendecomposition.
        
        For large spatial domains (n_points >> 1000), computing the full correlation
        matrix becomes prohibitively expensive. This method uses Nystroem approximation
        or other low-rank methods to approximate the correlation structure.
        
        Parameters:
            coords (Tensor): Spatial coordinates [n_points, n_dims]
            length_scale (float): Correlation length scale
            kernel (CorrelationKernel): Correlation kernel type
            n_eigenfunctions (int): Number of eigenfunctions to retain
            
        Returns:
            Tuple[Tensor, Tensor]: (eigenvalues, eigenvectors) for approximation
            
        Mathematical Expression:
            K ~= Sigma?^{n_eig} lambda? phi? phi?^T
            where lambda?, phi? are leading eigenvalues/eigenvectors
            
        References:
            - Williams, C. & Seeger, M. (2001). Using the Nystroem method to speed up
              kernel machines. NIPS 13.
        """
        n_points = coords.shape[0]
        
        if n_points <= n_eigenfunctions * 2:
            # Small enough to compute exactly
            distances = torch.norm(coords.unsqueeze(1) - coords.unsqueeze(0), dim=2)
            full_matrix = ParameterDistribution._kernel_values(distances, length_scale, kernel)
            eigenvals, eigenvecs = torch.linalg.eigh(full_matrix)
            
            # Sort in descending order and take top eigenfunction  
            sorted_indices = torch.argsort(eigenvals, descending=True)
            return eigenvals[sorted_indices[:n_eigenfunctions]], eigenvecs[:, sorted_indices[:n_eigenfunctions]]
        
        # Use Nystroem approximation for large matrices
        # Randomly sample subset of points
        subset_size = min(n_eigenfunctions * 4, n_points // 2)
        subset_indices = torch.randperm(n_points)[:subset_size]
        subset_coords = coords[subset_indices]
        
        # References:
        #   Williams and Seeger (2001), NIPS 13, eqs. 8-9.
        # Mathematical expression:
        #   K_mm U = U Lambda_m; lambda_n,i = (n / m) lambda_m,i;
        #   u_n,i proportional to K_nm u_m,i / lambda_m,i (unit-normalized).
        # Parameter meanings:
        #   K_mm: kernel matrix of the m subset points; K_nm: kernel between
        #   all n points and the subset (same kernel as requested); lambda,
        #   u: eigenvalues and eigenvectors.
        # Units:
        #   Dimensionless correlations; coordinates and length share a unit.
        # Domain/constraints:
        #   Every kept subset eigenvalue must be positive; otherwise
        #   ValueError (no eigenvalue clamping).
        # Sign convention:
        #   Eigenvector signs are arbitrary.
        subset_distances = torch.norm(subset_coords.unsqueeze(1) - subset_coords.unsqueeze(0), dim=2)
        K_subset = ParameterDistribution._kernel_values(subset_distances, length_scale, kernel)
        
        # Eigendecomposition of subset
        eigenvals_subset, eigenvecs_subset = torch.linalg.eigh(K_subset)
        
        # Cross-kernel between all points and the subset
        distances_ns = torch.norm(coords.unsqueeze(1) - subset_coords.unsqueeze(0), dim=2)  # [n, subset_size]
        K_ns = ParameterDistribution._kernel_values(distances_ns, length_scale, kernel)
        
        # Take top eigenfunctions
        sorted_indices = torch.argsort(eigenvals_subset, descending=True)
        n_keep = min(n_eigenfunctions, len(eigenvals_subset))
        kept = sorted_indices[:n_keep]
        kept_values = eigenvals_subset[kept]
        if not bool((kept_values > 0.0).all()):
            raise ValueError(
                f"Nystroem subset matrix has non-positive eigenvalues among the "
                f"{n_keep} kept modes (minimum {float(kept_values.min()):.3e}); "
                "reduce n_eigenfunctions"
            )

        # Nystroem extension of the kept eigenvectors, unit-normalized
        eigenvecs_full = K_ns @ (eigenvecs_subset[:, kept] / kept_values.unsqueeze(0))
        eigenvecs_full = eigenvecs_full / torch.norm(eigenvecs_full, dim=0, keepdim=True)
        
        return (kept_values * (n_points / subset_size), eigenvecs_full)


def sample_with_correlation(param_specs: Dict[str, DistributionSpec],
                           spatial_coords: Optional[Tensor] = None,
                           num_samples: int = 1,
                           device: torch.device = torch.device("cpu"),
                           seed: Optional[int] = None) -> Dict[str, Tensor]:
    """
    Convenience function for sampling parameters with spatial correlation.
    
    Parameters:
        param_specs (Dict[str, DistributionSpec]): Parameter distribution specs
        spatial_coords (Optional[Tensor]): Spatial coordinates for correlation
        num_samples (int): Number of sample sets
        device (torch.device): Computation device  
        seed (Optional[int]): Random seed
        
    Returns:
        Dict[str, Tensor]: Parameter samples with CF-1.8 compliant naming
        
    Example:
        >>> specs = {
        ...     'vcmax25': DistributionSpec(
        ...         dist_type=DistributionType.NORMAL,
        ...         params={'loc': 60.0, 'scale': 10.0},
        ...         bounds=(10.0, 200.0),
        ...         correlation_length=5000.0,  # 5km correlation
        ...         kernel=CorrelationKernel.EXPONENTIAL,
        ...         correlation_nugget=0.0,
        ...     )
        ... }
        >>> coords = torch.randn(100, 2) * 10000  # 100 points in 10km domain
        >>> samples = sample_with_correlation(specs, coords, num_samples=50)
        >>> print(samples['vcmax25'].shape)  # [50, 100]
    """
    param_dist = ParameterDistribution(param_specs, spatial_coords, device)
    
    return param_dist.sample(
        input_state=torch.empty(0), 
        num_samples=num_samples,
        seed=seed
    )
