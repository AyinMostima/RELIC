"""
Pedosphere numerical stability helpers.

This module contains the retained process-local flux limiting, pool-bound, and
stable-timestep utilities used by pedosphere transport Functors. Element budget
ownership belongs exclusively to StateManager and ConservationGuardian.

References:
- Manzoni, S. and Porporato, A. (2009). Soil carbon and nitrogen mineralization.
- Sierra, C. A. et al. (2012). Models of soil organic matter decomposition.
"""

import logging
import math
from typing import Any, Dict, Optional

import torch

from relic.parameters import get_global_manager

logger = logging.getLogger(__name__)


class NumericalStabilityController:
    """
    Ensures numerical stability for pedosphere processes.
    
    Implements adaptive time stepping, flux limiting, and pool constraints
    to maintain numerical stability.
    """
    
    def __init__(self,
                 min_pool_size: Optional[float] = None,
                 max_flux_fraction: Optional[float] = None,
                 cfl_number: Optional[float] = None,
                 min_time_step: Optional[float] = None,
                 max_time_step: Optional[float] = None,
                 parameter_manager: Optional[Any] = None):
        """
        Initialize stability controller.

        Parameters:
        -----------
        min_pool_size : Optional[float]
            Minimum allowed pool size [g m-2], defaults to parameter from manager
        max_flux_fraction : Optional[float]
            Maximum fraction of pool that can flux in one timestep, defaults to parameter from manager
        cfl_number : Optional[float]
            Courant-Friedrichs-Lewy number for stability, defaults to parameter from manager
        min_time_step : Optional[float]
            Minimum allowed timestep [s], defaults to parameter from manager
        max_time_step : Optional[float]
            Maximum allowed timestep [s], defaults to parameter from manager
        """
        def _require_param(key: str, units: Optional[str] = None) -> float:
            try:
                if parameter_manager is None:
                    value = get_global_manager().get(
                        key,
                        units=units,
                        expected_type=(float, int, torch.Tensor),
                    )
                else:
                    getter = getattr(parameter_manager, "get")
                    value = getter(
                        key,
                        units=units,
                        validate=units is not None,
                        expected_type=float,
                        required=True,
                    )
            except KeyError as exc:
                raise KeyError(
                    f"Required numerical stability parameter '{key}' is missing from ParameterManager."
                ) from exc
            except ValueError as exc:
                raise KeyError(
                    f"Required numerical stability parameter '{key}' is missing from ParameterManager."
                ) from exc
            if isinstance(value, torch.Tensor):
                return float(value.detach().reshape(-1)[0].tolist())
            return float(value)

        if min_pool_size is None:
            min_pool_size = _require_param('pedosphere.numerical.min_pool_size')

        if max_flux_fraction is None:
            max_flux_fraction = _require_param('pedosphere.numerical.max_flux_fraction')

        if cfl_number is None:
            cfl_number = _require_param('pedosphere.transport.cfl_number')

        if min_time_step is None:
            min_time_step = _require_param('pedosphere.numerical.min_time_step', units='s')

        if max_time_step is None:
            max_time_step = _require_param('pedosphere.numerical.max_time_step', units='s')

        self.min_pool_size = min_pool_size
        self.max_flux_fraction = max_flux_fraction
        self.cfl_number = cfl_number
        self.min_time_step = min_time_step
        self.max_time_step = max_time_step
        
        # Tracking
        self.flux_limiting_events = 0
        self.pool_clipping_events = 0
        self.timestep_reductions = 0
        
    def limit_fluxes(
        self,
        fluxes: Dict[str, torch.Tensor],
        pools: Dict[str, torch.Tensor],
        dt: float,
        *,
        source_pool_by_flux: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        """Limit donor-coupled outflows without relying on name heuristics.

        References:
        - Manzoni and Porporato (2009), Soil Biology and Biochemistry 41,
          1355-1367.
        - Sierra et al. (2012), Geoscientific Model Development 5, 1045-1060.

        Mathematical expression:
        - F_max,p = f_max P_p / dt.
        - s_p = min(1, F_max,p / sum_j |F_j,p|).
        - F_limited,j = s_p F_j,p.

        Parameter meanings:
        - F_j,p: requested outflow j from donor pool p [g m-2 s-1].
        - f_max: maximum removable pool fraction per step [1].
        - P_p: current donor stock [g m-2].
        - dt: positive integration interval [s].

        Every flux requires an explicit donor mapping. External boundary inputs
        must bypass this outflow limiter rather than using an implicit fallback.
        """
        dt_value = float(dt)
        if not math.isfinite(dt_value) or dt_value <= 0.0:
            raise ValueError("dt must be positive and finite for flux limiting")

        fraction = float(self.max_flux_fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("max_flux_fraction must be finite and within [0, 1]")

        flux_names = set(fluxes)
        mapping_names = set(source_pool_by_flux)
        if flux_names != mapping_names:
            missing = sorted(flux_names - mapping_names)
            extra = sorted(mapping_names - flux_names)
            raise ValueError(
                "source_pool_by_flux keys must exactly match fluxes; "
                f"missing={missing}, extra={extra}"
            )

        fluxes_by_pool: Dict[str, list[str]] = {}
        for flux_name, source_pool in source_pool_by_flux.items():
            if source_pool not in pools:
                raise KeyError(
                    f"Mapped source pool '{source_pool}' for flux "
                    f"'{flux_name}' is missing"
                )
            fluxes_by_pool.setdefault(source_pool, []).append(flux_name)

        limited_fluxes: Dict[str, torch.Tensor] = {}
        for source_pool, pool_flux_names in fluxes_by_pool.items():
            pool_value = pools[source_pool]
            reference_tensor = next(
                (
                    value
                    for value in (
                        pool_value,
                        *(fluxes[name] for name in pool_flux_names),
                    )
                    if torch.is_tensor(value)
                ),
                None,
            )
            if reference_tensor is None:
                reference_tensor = torch.tensor(0.0, dtype=torch.float64)

            device = reference_tensor.device
            dtype = reference_tensor.dtype
            pool_tensor = torch.as_tensor(
                pool_value,
                device=device,
                dtype=dtype,
            )
            flux_tensors = [
                torch.as_tensor(fluxes[name], device=device, dtype=dtype)
                for name in pool_flux_names
            ]
            try:
                broadcast = torch.broadcast_tensors(pool_tensor, *flux_tensors)
            except RuntimeError as exc:
                raise ValueError(
                    f"Fluxes mapped to '{source_pool}' are not broadcast-compatible "
                    "with the donor pool"
                ) from exc

            pool_broadcast = broadcast[0]
            flux_broadcast = list(broadcast[1:])
            if not bool(torch.isfinite(pool_broadcast).all().item()):
                raise ValueError(f"Source pool '{source_pool}' must be finite")
            if bool((pool_broadcast < 0).any().item()):
                raise ValueError(f"Source pool '{source_pool}' must be non-negative")
            for flux_name, flux_tensor in zip(pool_flux_names, flux_broadcast):
                if not bool(torch.isfinite(flux_tensor).all().item()):
                    raise ValueError(f"Flux '{flux_name}' must be finite")

            requested_total = torch.zeros_like(pool_broadcast)
            for flux_tensor in flux_broadcast:
                requested_total = requested_total + torch.abs(flux_tensor)
            maximum_total = pool_broadcast * fraction / dt_value
            scale = torch.ones_like(requested_total)
            positive_request = requested_total > 0
            safe_ratio = maximum_total / torch.where(
                positive_request,
                requested_total,
                torch.ones_like(requested_total),
            )
            scale = torch.where(
                positive_request,
                torch.minimum(scale, safe_ratio),
                scale,
            )

            for flux_name, flux_tensor in zip(pool_flux_names, flux_broadcast):
                limited_tensor = flux_tensor * scale
                if bool(torch.any(limited_tensor != flux_tensor).item()):
                    self.flux_limiting_events += 1
                    logger.debug(
                        "Limited flux %s from pool %s",
                        flux_name,
                        source_pool,
                    )
                original_flux = fluxes[flux_name]
                if torch.is_tensor(original_flux) or torch.is_tensor(pool_value):
                    limited_fluxes[flux_name] = limited_tensor
                else:
                    limited_fluxes[flux_name] = float(
                        limited_tensor.detach().reshape(-1)[0].item()
                    )

        return limited_fluxes
    def enforce_pool_bounds(self, pools: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Enforce minimum pool sizes and clip negative values.
        
        Parameters:
        -----------
        pools : Dict[str, torch.Tensor]
            Pool sizes [g m-2]
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Bounded pools
        """
        bounded_pools = {}
        
        for pool_name, pool_value in pools.items():
            is_tensor = torch.is_tensor(pool_value)
            pool_tensor = pool_value if is_tensor else torch.as_tensor(pool_value, dtype=torch.float32)
            if torch.any(pool_tensor < self.min_pool_size):
                self.pool_clipping_events += 1
                bounded_tensor = torch.clamp(pool_tensor, min=self.min_pool_size)
                bounded_scalar = bounded_tensor.detach().reshape(-1)[0].tolist()
                bounded_value = bounded_tensor if is_tensor else float(bounded_scalar)
                pool_mean = pool_tensor.detach().reshape(-1).mean().tolist()
                bounded_mean = bounded_tensor.detach().reshape(-1).mean().tolist()
                logger.debug(
                    "Clipped pool %s: %.2e -> %.2e",
                    pool_name,
                    float(pool_mean),
                    float(bounded_mean),
                )
            else:
                bounded_value = pool_value

            bounded_pools[pool_name] = bounded_value
        
        return bounded_pools
    
    def calculate_stable_timestep(self,
                                 decomposition_rates: Dict[str, torch.Tensor],
                                 pool_sizes: Dict[str, torch.Tensor]) -> float:
        """
        Calculate maximum stable timestep using CFL condition.
        
        Parameters:
        -----------
        decomposition_rates : Dict[str, torch.Tensor]
            Decomposition rate constants [s-1]
        pool_sizes : Dict[str, torch.Tensor]
            Current pool sizes [g m-2]
            
        Returns:
        --------
        float
            Maximum stable timestep [s]
            
        Mathematical Expression:
        -----------------------
        dt_max = CFL * min(1 / k_i) for all pools i
        """
        max_rate = None

        for rate_name, rate_value in decomposition_rates.items():
            rate_tensor = rate_value if torch.is_tensor(rate_value) else torch.as_tensor(rate_value)
            if rate_tensor.numel() == 0:
                continue
            rate_max = torch.max(torch.abs(rate_tensor))
            if max_rate is None:
                max_rate = rate_max
            else:
                max_rate = torch.maximum(max_rate, rate_max)

        # Avoid division by zero
        if max_rate is not None:
            max_rate_value = float(max_rate.detach().item())
            if max_rate_value > 0:
                dt_max = float(self.cfl_number) / max_rate_value
            else:
                dt_max = float(self.max_time_step)
        else:
            dt_max = float(self.max_time_step)
        
        # Apply reasonable bounds
        dt_max = min(dt_max, float(self.max_time_step))
        dt_max = max(dt_max, float(self.min_time_step))
        
        return dt_max
    
    def apply_implicit_correction(self,
                                 explicit_flux: torch.Tensor,
                                 rate_constant: torch.Tensor,
                                 pool_size: torch.Tensor,
                                 dt: float) -> torch.Tensor:
        """
        Apply semi-implicit correction for improved stability.
        
        Parameters:
        -----------
        explicit_flux : torch.Tensor
            Explicitly calculated flux [g m-2 s-1]
        rate_constant : torch.Tensor
            Rate constant [s-1]
        pool_size : torch.Tensor
            Current pool size [g m-2]
        dt : float
            Time step size [s]
            
        Returns:
        --------
        torch.Tensor
            Corrected flux
            
        Mathematical Expression:
        -----------------------
        Implicit: C_new = C_old / (1 + k * dt)
        Flux = k * C_new = k * C_old / (1 + k * dt)
        """
        damping_factor = 1.0 / (1.0 + rate_constant * dt)
        corrected_flux = explicit_flux * damping_factor
        
        return corrected_flux
    
    def get_statistics(self) -> Dict[str, float]:
        """Get stability control statistics."""
        return {
            'flux_limiting_events': self.flux_limiting_events,
            'pool_clipping_events': self.pool_clipping_events,
            'timestep_reductions': self.timestep_reductions
        }

__all__ = [
    "NumericalStabilityController",
]
