"""
Stochastic Weather Generation for Climate Forcing

This module implements stochastic weather generators for producing synthetic
meteorological time series consistent with RELIC's biogeochemical modeling
framework. It provides Ornstein-Uhlenbeck temperature processes and
Markov-Gamma precipitation models with explicit units and time metadata.

The weather generation follows established meteorological principles while
maintaining consistency with PyTorch for differentiable climate forcing
and uncertainty quantification.

Theory owner: docs/source/theory/stochastic_weather_generator.md

Mathematical Framework:
    Temperature (Ornstein-Uhlenbeck): dT = alpha(mu - T)dt + sigma dW
    Precipitation state (Markov): P(wet|wet) = p11, P(wet|dry) = p01  
    Precipitation amount (Gamma): I ~ Gamma(k, theta) on wet days

References:
    - Katz, R.W. (1996). Use of conditional stochastic models to generate 
      climate change scenarios. Clim. Change 32, 237-255.
    - Wilks, D.S. (2006). Statistical Methods in the Atmospheric Sciences. 
      Academic Press, 2nd edition.
    - Hasselmann, K. (1976). Stochastic climate models. Part I. Theory.
      Tellus 28, 473-485.
    - Gardiner, C. (2009). Stochastic Methods: A Handbook for the Natural
      and Social Sciences. Springer, 4th edition.

Authors: Yuchong Long
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import numpy as np
import torch
from torch import Tensor

from relic.core import device_policy
from relic.parameters import get_global_manager
from relic.utils.physical_constants import get_physical_constant
from relic.utils.time_constants import SECONDS_PER_DAY

from .category import ProcessMetadata, StochasticProcess

logger = logging.getLogger(__name__)

_TETENS_PREFIX = "relic.physical.constants.psychrometrics."


def _reject_options(options: Dict[str, object]) -> None:
    """The samplers take no free-form options; a misspelled one is an error."""
    if options:
        raise TypeError(f"Unsupported sample option(s): {sorted(options)}")


@contextmanager
def _seeded_draws(seed: Optional[int], device: torch.device) -> Iterator[None]:
    """Run the draws of one sample call on a generator state seeded with seed.

    The CPU generator (and the CUDA generator of a CUDA device) is forked and
    seeded locally, then restored, so the caller's global Torch RNG streams are
    unchanged. torch.manual_seed is not used: it would also reseed every CUDA
    generator outside the fork. With seed None the draws use the global streams.
    """
    if seed is None:
        yield
        return
    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield


_TEMPERATURE_KEYS = {
    "mean_temp": "stochastic.weather.temperature.mean_temp",
    "reversion_rate": "stochastic.weather.temperature.reversion_rate",
    "volatility": "stochastic.weather.temperature.volatility",
    "dt": "stochastic.weather.temperature.dt",
    "wet_day_temperature_offset": "stochastic.weather.temperature.wet_day_temperature_offset",
}
_PRECIPITATION_KEYS = {
    "p_wet_wet": "stochastic.weather.precipitation.p_wet_wet",
    "p_wet_dry": "stochastic.weather.precipitation.p_wet_dry",
    "gamma_shape": "stochastic.weather.precipitation.gamma_shape",
    "gamma_scale": "stochastic.weather.precipitation.gamma_scale",
    "initial_state": "stochastic.weather.precipitation.initial_state",
    "wet_day_threshold": "stochastic.weather.precipitation.wet_day_threshold",
}


@dataclass
class TemperatureParameters:
    """
    Parameters for Ornstein-Uhlenbeck temperature process.

    Parameters:
        mean_temp (float): Long-term (dry-day) mean temperature [K]
        reversion_rate (float): Mean reversion coefficient alpha [day-1]
        volatility (float): Volatility parameter sigma [K day-0.5]
        dt (float): Time step [day]
        wet_day_temperature_offset (float): Additive temperature offset on wet
            days used by WeatherGenerator, T = T_OU + offset * S [K]

    Every field is required; from_parameter_manager() reads the declared
    values of stochastic/weather_parameters.yaml (no code defaults). The
    initial temperature is the input_state of sample().

    Units:
        All parameters follow CF-1.8 conventions
        Temperature: Kelvin (K)
        Time: days
        Rate: day-1
    """
    mean_temp: float
    reversion_rate: float
    volatility: float
    dt: float
    wet_day_temperature_offset: float

    @classmethod
    def from_parameter_manager(cls) -> "TemperatureParameters":
        """Build the parameters from stochastic.weather.temperature.* in the YAML database."""
        manager = get_global_manager()
        return cls(**{
            name: float(manager.get(key, required=True)) for name, key in _TEMPERATURE_KEYS.items()
        })


@dataclass  
class PrecipitationParameters:
    """
    Parameters for Markov-Gamma precipitation model.
    
    Parameters:
        p_wet_wet (float): P(wet day | previous wet day) [dimensionless]
        p_wet_dry (float): P(wet day | previous dry day) [dimensionless]
        gamma_shape (float): Gamma distribution shape parameter k [dimensionless]
        gamma_scale (float): Gamma distribution scale parameter theta [mm day-1]
        initial_state (bool): Initial precipitation state (True=wet, False=dry)
        wet_day_threshold (float): Flux above which an observation is a wet day
            in log_probability [kg m-2 s-1]
    
    Mathematical Expressions:
        Transition probabilities: P_1_1 = p_wet_wet, P_0_1 = p_wet_dry
        Amount distribution: I ~ Gamma(k=gamma_shape, theta=gamma_scale)
        State evolution: S_{t+1} ~ Bernoulli(P_{S_t,1})
    """
    p_wet_wet: float
    p_wet_dry: float
    gamma_shape: float
    gamma_scale: float
    initial_state: bool
    wet_day_threshold: float

    @classmethod
    def from_parameter_manager(cls) -> "PrecipitationParameters":
        """Build the parameters from stochastic.weather.precipitation.* in the YAML database."""
        manager = get_global_manager()
        values = {name: manager.get(key, required=True) for name, key in _PRECIPITATION_KEYS.items()}
        initial_state = values.pop("initial_state")
        if not isinstance(initial_state, bool):
            raise TypeError(
                "stochastic.weather.precipitation.initial_state must be a boolean, "
                f"got {initial_state!r}"
            )
        return cls(initial_state=initial_state, **{name: float(v) for name, v in values.items()})


class OrnsteinUhlenbeckTemperature(StochasticProcess):
    """
    Ornstein-Uhlenbeck stochastic process for temperature modeling.
    
    This process implements the stochastic differential equation:
        dT(t) = alpha(mu - T(t))dt + sigmadW(t)
        
    where:
        alpha: mean reversion rate [day-1]
        mu: long-term mean temperature [K]
        sigma: volatility parameter [K day-1/2]
        W(t): Wiener process

    sample() uses the exact transition of the process (Gillespie 1996), not
    the Euler-Maruyama approximation:
        T_{t+Deltat} = mu + (T_t - mu) exp(-alpha Deltat)
                       + sigma sqrt((1 - exp(-2 alpha Deltat)) / (2 alpha)) xi

    where xi ~ N(0,1) is standard Gaussian noise; it is exact for every
    Deltat > 0.
    
    Mathematical Properties:
        - Mean-reverting: E[T(t)] -> mu as t -> inf
        - Gaussian process: T(t) ~ N(mu_t, sigma_t^2)
        - Stationary variance: sigma^2/(2alpha)
        - Autocorrelation: rho(s) = exp(-alpha|s|)
    
    References:
        - Hasselmann, K. (1976). Stochastic climate models. Tellus 28, 473-485.
        - Gardiner, C. (2009). Stochastic Methods. Springer, 4th edition.
        - Gillespie, D.T. (1996). Exact numerical simulation of the
          Ornstein-Uhlenbeck process and its integral. Phys. Rev. E 54,
          2084-2091.
    """

    def __init__(self, parameters: TemperatureParameters):
        """
        Initialize Ornstein-Uhlenbeck temperature process.
        
        Parameters:
            parameters (TemperatureParameters): Process parameters
        """
        self.params = parameters

        # Validate parameters
        if parameters.reversion_rate <= 0:
            raise ValueError("Reversion rate must be positive")
        if parameters.volatility <= 0:
            raise ValueError("Volatility must be positive") 
        if parameters.dt <= 0:
            raise ValueError("Time step must be positive")
            
        logger.info(f"Initialized OU temperature process: "
                   f"mean={parameters.mean_temp:.2f}K, "
                   f"alpha={parameters.reversion_rate:.3f} day-1, "
                   f"sigma={parameters.volatility:.2f}K day-0.5")

    def sample(self, input_state: Tensor, num_samples: int = 1,
               seed: Optional[int] = None, *, num_steps: int,
               **kwargs) -> Dict[str, Tensor]:
        """
        Generate temperature time series using Ornstein-Uhlenbeck process.

        Parameters:
            input_state (Tensor): Initial temperature [K], shape: (...)
            num_samples (int): Number of ensemble members
            seed (int, optional): Random seed for reproducibility
            num_steps (int): Number of time steps to simulate (required)
            **kwargs: none accepted; an unknown option raises TypeError
            
        Returns:
            Dict[str, Tensor]: Generated temperature time series
                'air_temperature': Temperature [K], shape: (num_samples, num_steps, ...)
                'temperature_anomaly': Anomaly from mean [K], same shape
                'temperature_trend': Deterministic component [K], same shape
                
        Mathematical Implementation:
            T_{n+1} = T_n*exp(-alpha*Deltat) + mu*(1-exp(-alpha*Deltat)) + sigma*sqrt((1-exp(-2alpha*Deltat))/(2alpha))*xi
            
        This is the exact solution avoiding numerical drift from Euler-Maruyama.
        """
        _reject_options(kwargs)
        # Ensure input_state is tensor
        if not isinstance(input_state, Tensor):
            input_state = torch.tensor(input_state, dtype=torch.float32)
            
        batch_shape = input_state.shape
        device = input_state.device
        dtype = input_state.dtype
        
        # Initialize output arrays
        temperatures = torch.zeros(num_samples, num_steps, *batch_shape, 
                                 device=device, dtype=dtype)
        anomalies = torch.zeros_like(temperatures)
        trends = torch.zeros_like(temperatures)
        
        # Process parameters
        alpha = self.params.reversion_rate
        mu = self.params.mean_temp
        sigma = self.params.volatility
        dt = self.params.dt
        
        # Exact solution coefficients
        exp_alpha_dt = torch.exp(torch.tensor(-alpha * dt))
        mean_coeff = 1.0 - exp_alpha_dt
        var_coeff = sigma * torch.sqrt((1.0 - torch.exp(torch.tensor(-2.0 * alpha * dt))) / (2.0 * alpha))
        
        with _seeded_draws(seed, device):
            for sample_idx in range(num_samples):
                # Initialize with input state
                current_temp = input_state.clone()

                for step in range(num_steps):
                    # Store current values
                    temperatures[sample_idx, step] = current_temp
                    anomalies[sample_idx, step] = current_temp - mu
                    trends[sample_idx, step] = mu + (input_state - mu) * torch.exp(torch.tensor(-alpha * step * dt))

                    # Generate next temperature using exact OU solution
                    noise = torch.randn_like(current_temp) * var_coeff
                    current_temp = current_temp * exp_alpha_dt + mu * mean_coeff + noise
        
        return {
            'air_temperature': temperatures,
            'temperature_anomaly': anomalies, 
            'temperature_trend': trends,
            'time_step': torch.full((num_samples, num_steps), dt),
            'reversion_rate': torch.full_like(temperatures, alpha),
            'volatility': torch.full_like(temperatures, sigma)
        }

    def log_probability(self, observations: Tensor, 
                       conditioning_state: Optional[Tensor] = None) -> Tensor:
        """
        Compute log-probability of temperature observations.
        
        Parameters:
            observations (Tensor): Observed temperatures [K]
            conditioning_state (Tensor, optional): Previous temperature state [K]
            
        Returns:
            Tensor: Log-probability values
            
        Mathematical Expression:
            For OU process: p(T_{t+dt}|T_t) ~ N(mu_t, sigma_t^2)
            where mu_t = T_t*exp(-alpha*dt) + mu*(1-exp(-alpha*dt))
                  sigma_t^2 = sigma^2*(1-exp(-2alpha*dt))/(2alpha)
        """
        if conditioning_state is None:
            # Marginal distribution (stationary); the variance is a Python float,
            # so its square root is taken as a tensor (torch.sqrt(float) raised).
            stationary_var = self.params.volatility**2 / (2 * self.params.reversion_rate)
            dist = torch.distributions.Normal(self.params.mean_temp,
                                            torch.tensor(stationary_var).sqrt())
        else:
            # Conditional distribution
            alpha_dt = self.params.reversion_rate * self.params.dt
            exp_alpha_dt = torch.exp(torch.tensor(-alpha_dt))
            
            cond_mean = (conditioning_state * exp_alpha_dt + 
                        self.params.mean_temp * (1 - exp_alpha_dt))
            cond_var = (self.params.volatility**2 * 
                       (1 - torch.exp(torch.tensor(-2 * alpha_dt))) / (2 * self.params.reversion_rate))
            
            dist = torch.distributions.Normal(cond_mean, torch.sqrt(cond_var))
            
        return dist.log_prob(observations)

    def moments(self) -> Dict[str, Tensor]:
        """
        Return theoretical moments of the OU process.
        
        Returns:
            Dict[str, Tensor]: Statistical moments
                'mean': Long-term mean temperature [K]
                'variance': Stationary variance [K^2]
                'std': Standard deviation [K]
                'autocorr_time': Correlation time [day]
        """
        alpha = self.params.reversion_rate
        sigma = self.params.volatility
        
        stationary_var = sigma**2 / (2 * alpha)
        autocorr_time = 1.0 / alpha
        
        return {
            'mean': torch.tensor(self.params.mean_temp),
            'variance': torch.tensor(stationary_var),
            'std': torch.tensor(stationary_var).sqrt(),
            'autocorr_time': torch.tensor(autocorr_time)
        }

    def describe(self) -> ProcessMetadata:
        """Return process metadata."""
        return ProcessMetadata(
            name="OrnsteinUhlenbeckTemperature",
            description="Stochastic temperature process with mean reversion",
            units={
                'air_temperature': 'K',
                'temperature_anomaly': 'K', 
                'temperature_trend': 'K',
                'time_step': 'day',
                'reversion_rate': 'day-1',
                'volatility': 'K day-0.5'
            },
            parameters={
                'mean_temp': f"{self.params.mean_temp} K",
                'reversion_rate': f"{self.params.reversion_rate} day-1", 
                'volatility': f"{self.params.volatility} K day-0.5",
                'dt': f"{self.params.dt} day"
            },
            references="Hasselmann (1976), Gardiner (2009)"
        )


class MarkovGammaPrecipitation(StochasticProcess):
    """
    Two-state Markov chain for precipitation occurrence with Gamma intensity.
    
    This model combines:
    1. Binary Markov chain for wet/dry state transitions
    2. Gamma distribution for precipitation amounts on wet days
    
    Mathematical Framework:
        State transitions:
            P(wet_{t+1} | wet_t) = p_1_1
            P(wet_{t+1} | dry_t) = p_0_1
            
        Precipitation intensity (wet days only):
            I_t ~ Gamma(k, theta) where k=shape, theta=scale
            
        Total precipitation:
            P_t = S_t * I_t where S_t in {0,1} is state
    
    References:
        - Katz, R.W. (1996). Conditional stochastic models. Clim. Change 32, 237-255.
        - Wilks, D.S. (2006). Statistical Methods in Atmospheric Sciences. Academic Press.
        - Richardson, C.W. (1981). Stochastic simulation of daily precipitation. Water Resour. Res. 17, 182-190.
    """
    
    def __init__(self, parameters: PrecipitationParameters):
        """
        Initialize Markov-Gamma precipitation model.
        
        Parameters:
            parameters (PrecipitationParameters): Model parameters
        """
        self.params = parameters

        # Validate parameters
        if not (0 <= parameters.p_wet_wet <= 1):
            raise ValueError("p_wet_wet must be in [0,1]")
        if not (0 <= parameters.p_wet_dry <= 1):
            raise ValueError("p_wet_dry must be in [0,1]")
        if parameters.gamma_shape <= 0:
            raise ValueError("Gamma shape must be positive")
        if parameters.gamma_scale <= 0:
            raise ValueError("Gamma scale must be positive")
        if not parameters.wet_day_threshold >= 0.0:
            raise ValueError("wet_day_threshold must be a non-negative flux")

        # Transition matrix
        self._p_matrix = torch.tensor([
            [1 - parameters.p_wet_dry, parameters.p_wet_dry],  # dry -> {dry, wet}
            [1 - parameters.p_wet_wet, parameters.p_wet_wet]   # wet -> {dry, wet} 
        ])
        
        # References: Wilks (2011), Statistical Methods in the Atmospheric
        #     Sciences, 3rd ed., Sect. 9.2.2 (two-state first-order Markov chain).
        # Mathematical expression: pi_wet = p01 / (p01 + p10) = p01 / (1 + p01 - p11).
        # Parameter meanings: p01 = P(wet | previous dry) = p_wet_dry,
        #     p11 = P(wet | previous wet) = p_wet_wet, p10 = 1 - p11.
        # Units: all probabilities are dimensionless.
        # Domain/constraints: the stationary distribution is unique only when
        #     p01 + p10 > 0; for p01 = 0 and p11 = 1 both states are absorbing,
        #     so no stationary probability is stored (None) and a request for
        #     stationary quantities raises. The former denominator 2 - p01 - p11
        #     and the 0.5 substitute were not the chain's stationary law.
        # Sign convention: not applicable (probability).
        p01, p11 = parameters.p_wet_dry, parameters.p_wet_wet
        denominator = 1.0 + p01 - p11
        self._p_wet_steady: Optional[float] = p01 / denominator if denominator > 0.0 else None
        
        logger.info(f"Initialized Markov-Gamma precipitation: "
                   f"p11={p11:.3f}, p01={p01:.3f}, "
                   f"Gamma(k={parameters.gamma_shape:.2f}, theta={parameters.gamma_scale:.2f})")

    def sample(self, input_state: Tensor, num_samples: int = 1,
               seed: Optional[int] = None, *, num_steps: int,
               **kwargs) -> Dict[str, Tensor]:
        """
        Generate precipitation time series using Markov-Gamma model.

        Parameters:
            input_state (Tensor): Not used for precipitation (stateless start)
            num_samples (int): Number of ensemble members
            seed (int, optional): Random seed for reproducibility
            num_steps (int): Number of time steps to simulate (required)
            **kwargs: none accepted; an unknown option raises TypeError
            
        Returns:
            Dict[str, Tensor]: Generated precipitation data
                'precipitation_flux': Precipitation [kg m-2 s-1], shape: (num_samples, num_steps)
                'precipitation_state': Binary wet/dry state [dimensionless], same shape
                'wet_spell_length': Length of current wet spell [day], same shape
                'dry_spell_length': Length of current dry spell [day], same shape
                
        Mathematical Implementation:
            State: S_{t+1} ~ Bernoulli(P_{S_t,1})
            Intensity: I_t ~ Gamma(k, theta) if S_t = 1, else I_t = 0
        """
        _reject_options(kwargs)
        device = device_policy.resolve_device(
            explicit_device=(input_state.device if isinstance(input_state, Tensor) else None),
            module_name=self.__class__.__module__,
            default="auto",
        )
        
        # Initialize outputs
        precip_flux = torch.zeros(num_samples, num_steps, device=device)
        precip_state = torch.zeros(num_samples, num_steps, device=device)
        wet_lengths = torch.zeros(num_samples, num_steps, device=device)
        dry_lengths = torch.zeros(num_samples, num_steps, device=device)
        
        # Gamma distribution for intensities
        gamma_dist = torch.distributions.Gamma(self.params.gamma_shape, 
                                             rate=1.0/self.params.gamma_scale)
        # References: Richardson (1981); CF precipitation flux units.
        # Mathematical expression: q = P / S_d.
        # Parameter meanings: P is daily water depth and S_d is seconds per day.
        # Units: P [mm day-1] is water-equivalent kg m-2 day-1; q [kg m-2 s-1].
        # Domain/constraints: P >= 0 and S_d > 0 from the physical-constant SSOT.
        # Sign convention: positive q is downward precipitation input.
        seconds_per_day = float(SECONDS_PER_DAY)
        
        # The Gamma intensities and the Bernoulli transitions are drawn on the
        # CPU generator (the outputs may live on another device).
        with _seeded_draws(seed, torch.device("cpu")):
            for sample_idx in range(num_samples):
                # Initialize state
                current_state = int(self.params.initial_state)
                wet_spell = 0 if current_state == 0 else 1
                dry_spell = 1 if current_state == 0 else 0

                for step in range(num_steps):
                    # Update spell lengths
                    if current_state == 1:  # wet
                        wet_spell += 1
                        dry_spell = 0
                    else:  # dry
                        dry_spell += 1
                        wet_spell = 0

                    # Store current state and spell lengths
                    precip_state[sample_idx, step] = current_state
                    wet_lengths[sample_idx, step] = wet_spell
                    dry_lengths[sample_idx, step] = dry_spell

                    # Generate precipitation amount
                    if current_state == 1:  # wet day
                        # Sample from Gamma distribution [mm day-1]
                        precip_mm = gamma_dist.sample()
                        # Convert to kg m-2 s-1 (CF-1.8 standard)
                        precip_flux[sample_idx, step] = precip_mm / seconds_per_day
                    else:  # dry day
                        precip_flux[sample_idx, step] = 0.0

                    # Transition to next state
                    if current_state == 1:  # currently wet
                        prob_wet = self.params.p_wet_wet
                    else:  # currently dry
                        prob_wet = self.params.p_wet_dry

                    current_state = int(torch.rand(1).item() < prob_wet)

        return {
            'precipitation_flux': precip_flux,
            'precipitation_state': precip_state,
            'wet_spell_length': wet_lengths,
            'dry_spell_length': dry_lengths,
            'transition_probs': self._p_matrix.expand(num_samples, num_steps, 2, 2)
        }

    def log_probability(self, observations: Tensor,
                       conditioning_state: Optional[Tensor] = None) -> Tensor:
        """
        Compute log-probability for precipitation observations.
        
        Parameters:
            observations (Tensor): Observed precipitation [kg m-2 s-1]
            conditioning_state (Tensor, optional): Previous precipitation state [0/1]
            
        Returns:
            Tensor: Log-probability values
            
        Mathematical Expression:
            P(P_t | S_{t-1}) = P(S_t | S_{t-1}) * P(I_t | S_t)
            where P(I_t | S_t=1) ~ Gamma(k,theta), P(I_t | S_t=0) = delta(I_t)
        """
        # References: Richardson (1981); CF precipitation flux units.
        # Mathematical expression: P = q * S_d.
        # Parameter meanings: q is observed flux and S_d is seconds per day.
        # Units: q [kg m-2 s-1] maps to P [mm day-1] water equivalent.
        # Domain/constraints: q >= 0 and S_d > 0 from the physical-constant SSOT.
        # Sign convention: positive q is downward precipitation input.
        seconds_per_day = float(SECONDS_PER_DAY)
        obs_mm_day = observations * seconds_per_day
        
        # Separate wet and dry observations at the declared wet-day threshold
        # (stochastic.weather.precipitation.wet_day_threshold).
        is_wet = (observations > self.params.wet_day_threshold)
        is_dry = ~is_wet
        
        log_prob = torch.zeros_like(observations)
        
        # Gamma distribution for wet days
        gamma_dist = torch.distributions.Gamma(self.params.gamma_shape,
                                             rate=1.0/self.params.gamma_scale)
        
        if conditioning_state is None:
            # Marginal probability (stationary wet probability for every value;
            # a Python float here made torch.log fail)
            p_wet = torch.full_like(observations, self._require_stationary_wet_probability())
        else:
            # Conditional on previous state
            prev_wet = (conditioning_state > 0.5)
            p_wet = torch.where(prev_wet, self.params.p_wet_wet, self.params.p_wet_dry).to(
                observations.dtype
            )

        # Wet day probabilities
        if is_wet.any():
            log_prob[is_wet] = torch.log(p_wet[is_wet]) + gamma_dist.log_prob(obs_mm_day[is_wet])

        # Dry day probabilities
        if is_dry.any():
            log_prob[is_dry] = torch.log(1 - p_wet[is_dry])
        
        return log_prob

    def _require_stationary_wet_probability(self) -> float:
        """Stationary wet probability, or ValueError when the chain has none."""
        if self._p_wet_steady is None:
            raise ValueError(
                "The wet/dry chain with p_wet_dry = 0 and p_wet_wet = 1 has two absorbing "
                "states and no unique stationary distribution"
            )
        return self._p_wet_steady

    def moments(self) -> Dict[str, Tensor]:
        """
        Return theoretical moments of the Markov-Gamma process.
        
        Returns:
            Dict[str, Tensor]: Statistical moments including transition probabilities
        """
        p_wet = self._require_stationary_wet_probability()
        gamma_mean = self.params.gamma_shape * self.params.gamma_scale  # mm day-1
        gamma_var = self.params.gamma_shape * self.params.gamma_scale**2  # (mm day-1)^2
        
        # References: Katz (1996); Richardson (1981); CF precipitation units.
        # Mathematical expression: E[q] = E[P]/S_d; Var(q) = Var(P)/S_d^2.
        # Parameter meanings: P is daily amount, q is SI flux, S_d is seconds/day.
        # Units: E[q] [kg m-2 s-1] and Var(q) [(kg m-2 s-1)^2].
        # Domain/constraints: Gamma moments are finite and S_d > 0 from SSOT.
        # Sign convention: positive moments describe downward precipitation input.
        seconds_per_day = float(SECONDS_PER_DAY)
        mean_precip = p_wet * gamma_mean / seconds_per_day
        var_precip = (
            p_wet * (gamma_var + (1 - p_wet) * gamma_mean**2)
            / seconds_per_day**2
        )
        
        return {
            'wet_probability': torch.tensor(p_wet),
            'mean_precipitation': torch.tensor(mean_precip),
            'variance_precipitation': torch.tensor(var_precip),
            'mean_wet_intensity': torch.tensor(gamma_mean / seconds_per_day),
            'gamma_shape': torch.tensor(self.params.gamma_shape),
            'gamma_scale': torch.tensor(self.params.gamma_scale / seconds_per_day)
        }

    def describe(self) -> ProcessMetadata:
        """Return process metadata."""
        return ProcessMetadata(
            name="MarkovGammaPrecipitation",
            description="Two-state Markov chain with Gamma precipitation intensities",
            units={
                'precipitation_flux': 'kg m-2 s-1',
                'precipitation_state': '1',
                'wet_spell_length': 'day',
                'dry_spell_length': 'day'
            },
            parameters={
                'p_wet_wet': f"{self.params.p_wet_wet}",
                'p_wet_dry': f"{self.params.p_wet_dry}",
                'gamma_shape': f"{self.params.gamma_shape}",
                'gamma_scale': f"{self.params.gamma_scale} mm day-1"
            },
            references="Katz (1996), Wilks (2006), Richardson (1981)"
        )


class WeatherGenerator(StochasticProcess):
    """
    Coupled stochastic weather generator combining temperature and precipitation.
    
    This class coordinates multiple weather processes to generate consistent
    meteorological forcing for biogeochemical simulations. It ensures temporal
    coherence between different meteorological variables while maintaining
    their individual stochastic properties.
    
    Features:
        - Ornstein-Uhlenbeck temperature
        - Markov-Gamma precipitation with realistic spell statistics
        - Wet-day temperature offset (temperature conditioned on the wet/dry
          state, Richardson 1981)
        - CF-1.8 compliant output with proper units and metadata
        - PyTorch integration for differentiable weather generation
    
    References:
        - Kilsby, C.G. et al. (2007). A daily weather generator. J. Hydrol. 329, 1-18.
        - Srikanthan, R. & McMahon, T.A. (2001). Stochastic generation of annual, 
          monthly and daily climate data. J. Hydrol. 245, 1-26.
    """
    
    def __init__(self,
                 temp_params: TemperatureParameters,
                 precip_params: PrecipitationParameters):
        """
        Initialize coupled weather generator.

        Parameters:
            temp_params (TemperatureParameters): Temperature process parameters,
                including the wet-day temperature offset [K]
            precip_params (PrecipitationParameters): Precipitation process parameters
        """
        # The Markov-Gamma chain advances one day per step (daily amounts in
        # mm day-1); the temperature recursion shares its step index.
        if temp_params.dt != 1.0:
            raise ValueError(
                "WeatherGenerator couples the temperature recursion with the daily "
                f"Markov-Gamma chain; temperature dt must be one day, got {temp_params.dt} day"
            )
        self.temp_gen = OrnsteinUhlenbeckTemperature(temp_params)
        self.precip_gen = MarkovGammaPrecipitation(precip_params)
        self.wet_day_temperature_offset = float(temp_params.wet_day_temperature_offset)

        logger.info(
            "Initialized WeatherGenerator with wet_day_temperature_offset=%.3f K",
            self.wet_day_temperature_offset,
        )

    def sample(self, input_state: Tensor, num_samples: int = 1,
               seed: Optional[int] = None, *, num_steps: int,
               **kwargs) -> Dict[str, Tensor]:
        """
        Generate coupled weather time series.

        Parameters:
            input_state (Tensor): Initial temperature [K]
            num_samples (int): Number of ensemble members
            seed (int, optional): Random seed for reproducibility
            num_steps (int): Number of daily steps to simulate (required)
            **kwargs: none accepted; an unknown option raises TypeError

        Returns:
            Dict[str, Tensor]: Complete weather forcing dataset
                - All temperature variables from OrnsteinUhlenbeckTemperature
                - All precipitation variables from MarkovGammaPrecipitation
                - saturation_vapor_pressure [Pa] from the generated temperature
        """
        _reject_options(kwargs)
        if seed is not None:
            # The temperature and precipitation seeds are two children of one
            # numpy SeedSequence: independent streams for every member seed. A
            # fixed offset (seed + 1000) made member s reuse the temperature
            # stream of member s + 1000 for its precipitation.
            temp_sequence, precip_sequence = np.random.SeedSequence(seed).spawn(2)
            temp_seed = int(temp_sequence.generate_state(1)[0])
            precip_seed = int(precip_sequence.generate_state(1)[0])
        else:
            temp_seed = precip_seed = None
            
        # Generate temperature
        temp_data = self.temp_gen.sample(
            input_state, num_samples, temp_seed, num_steps=num_steps
        )
        
        # Generate precipitation  
        precip_data = self.precip_gen.sample(
            input_state, num_samples, precip_seed, num_steps=num_steps
        )

        # Condition the temperature on the wet/dry state of the same day
        if self.wet_day_temperature_offset != 0.0:
            temp_data = self._apply_wet_day_offset(temp_data, precip_data)
        
        # Combine outputs
        weather_data = {**temp_data, **precip_data}
        
        # Add derived variables
        weather_data.update(self._compute_derived_variables(weather_data))
        
        return weather_data

    def _apply_wet_day_offset(self, temp_data: Dict[str, Tensor],
                              precip_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Shift the temperature of wet days by the declared offset.

        References:
            Richardson, C.W. (1981). Water Resour. Res. 17, 182-190 (WGEN:
            daily temperature conditioned on the wet/dry state).
        Mathematical expression:
            T = T_OU + delta_wet * S, S in {0, 1}.
        Parameter meanings:
            T_OU is the Ornstein-Uhlenbeck temperature (mean_temp is the
            dry-day mean), delta_wet the wet-day temperature offset, S the
            wet/dry state of the same day.
        Units:
            T, T_OU and delta_wet in K; S dimensionless.
        Domain/constraints:
            An additive shift of the daily mean; the temperature marginal
            becomes a two-component mixture and no correlation coefficient is
            implied or reported.
        Sign convention:
            delta_wet < 0 cools wet days.
        """
        adjustment = self.wet_day_temperature_offset * precip_data['precipitation_state'].to(
            temp_data['air_temperature'].dtype
        )
        while adjustment.dim() < temp_data['air_temperature'].dim():
            adjustment = adjustment.unsqueeze(-1)
        temp_data['air_temperature'] = temp_data['air_temperature'] + adjustment
        temp_data['temperature_anomaly'] = temp_data['temperature_anomaly'] + adjustment
        return temp_data

    def _compute_derived_variables(self, weather_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Compute additional meteorological variables.
        
        Parameters:
            weather_data (Dict[str, Tensor]): Primary weather variables
            
        Returns:
            Dict[str, Tensor]: Derived meteorological variables
        """
        # The canonical psychrometrics owner loads the scientific package, so it
        # is imported on first use rather than with the stochastic package.
        from relic.scientific.hydrosphere.utils.psychrometrics import (
            compute_saturation_vapor_pressure,
        )

        temp = weather_data['air_temperature']  # [K]
        kelvin_offset = float(get_physical_constant("celsius_to_kelvin"))

        # References: Tetens (1930); Allen et al. (1998) FAO-56, Eq. 11.
        # Mathematical expression: e_s = e_0 exp(a T_c / (T_c + b)), T_c = T - T_0.
        # Parameter meanings: e_0 = saturation vapour pressure at 0 degC, a =
        #     Tetens coefficient, b = Tetens temperature offset, T = air
        #     temperature, T_0 = 273.15 K from the physical-constant SSOT.
        # Units: e_0 and e_s in Pa, a dimensionless, b and T_c in degC, T in K.
        # Domain/constraints: liquid-water formulation; T_c + b > 0. The Tetens
        #     constants are read from the physical-constant SSOT (the former
        #     inline form used T in K in the denominator and local literals).
        # Sign convention: e_s > 0 and increases with temperature.
        e_sat = compute_saturation_vapor_pressure(
            temp,
            tetens_base_pressure=float(get_physical_constant(_TETENS_PREFIX + "tetens_base_pressure")),
            tetens_coeff_a=float(get_physical_constant(_TETENS_PREFIX + "tetens_coefficient_a")),
            tetens_coeff_b=float(get_physical_constant(_TETENS_PREFIX + "tetens_coefficient_b")),
            kelvin_offset=kelvin_offset,
        )

        # No relative humidity or vapour pressure deficit is produced: WGEN
        # (Richardson 1981) generates precipitation, temperature and radiation
        # only (the former RH formula used invented coefficients). The WXGEN
        # humidity model (Sharpley & Williams 1990; SWAT 2005 theory Sect.
        # 1:3.5) needs a site dew-point climatology and, on wet days, gives
        # values >= 1 that its reference code clips, so it is not implemented.
        return {
            'saturation_vapor_pressure': e_sat  # [Pa]
        }

    def describe(self) -> ProcessMetadata:
        """Return combined weather generator metadata."""
        return ProcessMetadata(
            name="WeatherGenerator", 
            description="Coupled stochastic weather generator (Temperature + Precipitation)",
            units={
                **self.temp_gen.describe().units,
                **self.precip_gen.describe().units,
                'saturation_vapor_pressure': 'Pa'
            },
            parameters={
                'wet_day_temperature_offset': f"{self.wet_day_temperature_offset} K",
                'temperature': self.temp_gen.describe().parameters,
                'precipitation': self.precip_gen.describe().parameters
            },
            references="Kilsby et al. (2007), Srikanthan & McMahon (2001)"
        )

    def log_probability(self, observations: Tensor, 
                       conditioning_state: Optional[Tensor] = None) -> Tensor:
        """
        Compute the joint log-probability of temperature and precipitation.

        log p(T, P | T_prev, S_prev) = log p_OU(T | T_prev) + log p_MG(P | S_prev)
        holds for independent components, i.e. without the wet-day
        temperature offset (wet_day_temperature_offset == 0); with an offset the joint
        density is not implemented and a request raises.

        Parameters:
            observations (Tensor): [..., 2] stacking [air_temperature [K],
                precipitation_flux [kg m-2 s-1]] on the last axis
            conditioning_state (Tensor, optional): [..., 2] stacking the previous
                [air_temperature [K], wet/dry state [0/1]]; None uses the
                stationary marginals

        Returns:
            Tensor: Combined log-probability
        """
        if self.wet_day_temperature_offset != 0.0:
            raise NotImplementedError(
                "The joint log-probability with a wet_day_temperature_offset is not "
                "implemented; the independent sum would misstate it."
            )
        if observations.shape[-1] != 2:
            raise ValueError(
                "observations must stack [air_temperature, precipitation_flux] on the "
                f"last axis, got shape {tuple(observations.shape)}"
            )
        if conditioning_state is not None and conditioning_state.shape[-1] != 2:
            raise ValueError(
                "conditioning_state must stack [air_temperature, wet/dry state] on the "
                f"last axis, got shape {tuple(conditioning_state.shape)}"
            )
        temp_condition = None if conditioning_state is None else conditioning_state[..., 0]
        precip_condition = None if conditioning_state is None else conditioning_state[..., 1]
        temp_log_prob = self.temp_gen.log_probability(observations[..., 0], temp_condition)
        precip_log_prob = self.precip_gen.log_probability(observations[..., 1], precip_condition)
        return temp_log_prob + precip_log_prob

    def moments(self) -> Dict[str, Tensor]:
        """
        Return theoretical moments of the combined weather process.
        
        Returns:
            Dict[str, Tensor]: Combined moments from temperature and precipitation
        """
        temp_moments = self.temp_gen.moments()
        precip_moments = self.precip_gen.moments()
        offset = self.wet_day_temperature_offset

        if offset != 0.0:
            # References: Richardson (1981) (wet-day conditioning); moments of a
            #     sum of independent components.
            # Mathematical expression: T = T_OU + delta S, S ~ Bernoulli(pi_wet)
            #     at stationarity, E[T] = mu + delta pi_wet,
            #     Var[T] = sigma^2 / (2 alpha) + delta^2 pi_wet (1 - pi_wet).
            # Parameter meanings: mu, alpha, sigma the OU mean, reversion rate and
            #     volatility; delta the wet-day offset; pi_wet the stationary wet
            #     probability of the Markov chain.
            # Units: E[T] in K, Var[T] in K2; pi_wet dimensionless.
            # Domain/constraints: the OU temperature and the chain are independent;
            #     the autocorrelation of the mixture is not a single exponential,
            #     so no autocorrelation time is reported.
            # Sign convention: delta < 0 lowers the mean.
            pi_wet = self.precip_gen._require_stationary_wet_probability()
            variance = temp_moments['variance'] + offset**2 * pi_wet * (1.0 - pi_wet)
            temp_moments = {
                'mean': temp_moments['mean'] + offset * pi_wet,
                'variance': variance,
                'std': variance.sqrt(),
            }

        combined_moments = {f'temperature_{key}': value for key, value in temp_moments.items()}
        combined_moments.update(
            {f'precipitation_{key}': value for key, value in precip_moments.items()}
        )
        combined_moments['wet_day_temperature_offset'] = torch.tensor(offset)
        return combined_moments
