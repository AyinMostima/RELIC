"""
Nitrogen-15 Isotope Fractionation Calculator

This module provides the 15N isotope fractionation calculator that wraps the
existing fractionation.py module to provide consistency with the new 
multi-isotope architecture.

The calculator implements 15N fractionation for nitrogen cycle processes:
- Mineralization (orgN -> NH4+)
- Nitrification (NH4+ -> NO3-)  
- Denitrification (NO3- -> N2O/N2)

Key Features:
- Wraps existing IsotopeFractionationCalculator from fractionation.py
- Provides consistency with multi-isotope base class
- Maintains all original functionality and scientific accuracy
- Temperature-dependent Q10 and Arrhenius models
- Natural abundance ratio conversions

Mathematical Framework:
- delta15N = (Rsample/Rair_N2 - 1) * 1000?
- Air N2 standard: R = 0.0036765
- Fractionation processes follow established literature values

References:
- Mariotti et al. (1981). Experimental determination of nitrogen kinetic isotope fractionation
- Casciotti et al. (2003). Linking diversity and stable isotope fractionation
- All references from the original fractionation.py module
"""

import torch
from typing import Dict, Optional, Union, Any, List
import warnings

from relic.parameters import UnifiedParameterManager

from .base import IsotopeFractionationCalculator, ProcessNotSupportedError
from ..config.models import Nitrogen15Config, FractionationModelType


class Nitrogen15FractionationCalculator(IsotopeFractionationCalculator):
    """
    15N isotope fractionation calculator.
    
    Provides 15N fractionation calculations for nitrogen cycle processes
    by wrapping the existing validated fractionation calculator.
    
    Supported Processes:
    - mineralization: orgN -> NH4+ (Q10 model, alpha_0=0.995, Q10=2.2)  
    - nitrification: NH4+ -> NO3- (Arrhenius model, alpha_0=0.985, Ea=50 kJ/mol)
    - denitrification: NO3- -> N2O/N2 (Q10 model, alpha_0=0.980, Q10=1.8)
    """
    
    def __init__(
        self,
        config: Optional[Nitrogen15Config] = None,
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        """
        Initialize 15N fractionation calculator.
        
        Parameters:
            config: Isotope configuration (uses default if None)
        """
        super().__init__("15N", parameter_manager=parameter_manager)
        
        # Air N2 standard for 15N
        self.air_n2_standard = float(
            self.get_parameter_value(
                "standards.nitrogen_15.air_ratio",
                namespace="isotopes",
                required=True)
        )
        # Alias for test consistency
        self.atmospheric_standard = self.air_n2_standard
        self.ratio_bounds = (
            float(
                self.get_parameter_value(
                    "ratio_min",
                    namespace="isotopes.fractionation",
                    required=True)
            ),
            float(
                self.get_parameter_value(
                    "ratio_max",
                    namespace="isotopes.fractionation",
                    required=True)
            ),
        )
        
        # Create or use provided configuration
        if config is None:
            config = Nitrogen15Config()
        self.config = config
        
        # Store fractionation parameters
        self.process_configs = config.processes
        self.enabled_processes = [
            name for name, proc_config in self.process_configs.items() 
            if proc_config.enabled
        ]
        
        # Process mapping
        self.supported_process_names = list(config.processes.keys())
        # Alias for test consistency
        self.supported_processes = self.supported_process_names
        
        self.logger.info(f"Initialized 15N fractionation calculator with processes: {self.supported_process_names}")
    
    def calculate_fractionation_factor(
        self,
        process_type: Union[str, Any],
        temperature: torch.Tensor,
        process_params: Optional[Any] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 15N fractionation factor for nitrogen cycle processes.
        
        Parameters:
            process_type: Process name (str) or type
            temperature: Temperature [K], shape [...]
            process_params: Unused (parameters from configuration)
            environmental_state: Environmental variables (unused)
            **kwargs: Additional parameters
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
            
        Raises:
            ProcessNotSupportedError: If process not supported
        """
        # Convert to string if needed
        if not isinstance(process_type, str):
            process_type = str(process_type)
        
        # Validate process
        if process_type not in self.supported_process_names:
            raise ProcessNotSupportedError(
                f"Process '{process_type}' not supported for 15N. "
                f"Supported processes: {self.supported_process_names}"
            )
        
        # Validate temperature
        temperature = self.validate_temperature(temperature, process_type)
        
        if process_type not in self.process_configs:
            raise ProcessNotSupportedError(f"Process {process_type} not configured")

        process_config = self.process_configs[process_type]
        frac_params = process_config.fractionation

        if frac_params.model_type == FractionationModelType.Q10:
            alpha = self._compute_q10_fractionation(
                temperature, frac_params.alpha_0, frac_params.Q10,
                frac_params.T_ref, process_type
            )
        elif frac_params.model_type == FractionationModelType.ARRHENIUS:
            alpha = self._compute_arrhenius_fractionation(
                temperature, frac_params.alpha_0, frac_params.E_a,
                frac_params.T_ref, process_type
            )
        else:
            alpha = torch.full_like(temperature, frac_params.alpha_0)

        return self.validate_fractionation_factor(alpha, process_type)
    
    def delta_notation(self, isotope_ratio: torch.Tensor) -> torch.Tensor:
        """
        Convert 15N/14N ratio to delta notation.
        
        delta15N = (Rsample/Rair_N2 - 1) * 1000?
        
        Parameters:
            isotope_ratio: 15N/14N ratio [dimensionless]
            
        Returns:
            Delta value delta15N [?]
        """
        return (isotope_ratio / self.air_n2_standard - 1.0) * 1000.0
    
    def ratio_from_delta(self, delta_value: torch.Tensor) -> torch.Tensor:
        """
        Convert delta notation to 15N/14N ratio.
        
        Rsample = Rair_N2 * (delta15N/1000 + 1)
        
        Parameters:
            delta_value: Delta value delta15N [?]
            
        Returns:
            Isotope ratio 15N/14N [dimensionless]
        """
        return self.air_n2_standard * (delta_value / 1000.0 + 1.0)
    
    def get_supported_processes(self) -> List[str]:
        """Get list of supported nitrogen cycle processes."""
        return self.supported_process_names.copy()
    
    def get_process_parameters(self, process_name: str) -> Dict[str, Any]:
        """
        Get fractionation parameters for a specific process.
        
        Parameters:
            process_name: Name of nitrogen process
            
        Returns:
            Dictionary with process parameters
        """
        if process_name not in self.config.processes:
            raise ProcessNotSupportedError(f"Process '{process_name}' not configured")
        
        process_config = self.config.processes[process_name]
        frac_params = process_config.fractionation
        
        params = {
            'model_type': frac_params.model_type,
            'alpha_0': frac_params.alpha_0,
            'T_ref': frac_params.T_ref,
            'T_min': frac_params.T_min,
            'T_max': frac_params.T_max,
            'source_pool': process_config.source_pool,
            'target_pool': process_config.target_pool,
            'flux_variable': process_config.flux_variable,
            'enabled': process_config.enabled
        }
        
        # Add model-specific parameters
        if frac_params.model_type == FractionationModelType.Q10:
            params['Q10'] = frac_params.Q10
        elif frac_params.model_type == FractionationModelType.ARRHENIUS:
            params['E_a'] = frac_params.E_a
        
        return params
    
    def get_natural_abundance(self) -> float:
        """Get natural 15N abundance ratio."""
        return self.air_n2_standard
    
    def validate_isotope_ratio(self, ratio: torch.Tensor) -> torch.Tensor:
        """
        Validate 15N abundance ratio bounds.
        
        Parameters:
            ratio: 15N abundance ratio [dimensionless]
            
        Returns:
            Validated ratio [dimensionless]
        """
        if not isinstance(ratio, torch.Tensor):
            ratio = torch.tensor(ratio, dtype=torch.float32)
        
        # Check bounds (0 to 1 for abundance ratio)
        if torch.any(ratio < 0) or torch.any(ratio > 1):
            ratio_min_value = ratio.min().detach().reshape(-1)[0].tolist()
            ratio_max_value = ratio.max().detach().reshape(-1)[0].tolist()
            warnings.warn(
                f"15N abundance ratio outside valid range [0, 1]: "
                f"[{ratio_min_value:.6f}, {ratio_max_value:.6f}], clamping"
            )
            ratio = torch.clamp(
                ratio,
                min=self.ratio_bounds[0],
                max=self.ratio_bounds[1],
            )
        
        # Check for extreme values (>10x natural abundance is unusual)
        natural_abundance = self.get_natural_abundance()
        if torch.any(ratio > 10 * natural_abundance):
            ratio_max_value = ratio.max().detach().reshape(-1)[0].tolist()
            warnings.warn(
                f"15N abundance ratio >10x natural abundance detected: "
                f"max = {ratio_max_value:.6f}, natural = {natural_abundance:.6f}"
            )
        
        return ratio
    
    def compute_bulk_fractionation_factors(
        self,
        temperature: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute fractionation factors for all supported processes.
        
        Parameters:
            temperature: Temperature [K]
            
        Returns:
            Dictionary mapping process names to fractionation factors
        """
        fractionation_factors = {}
        for process_name in self.supported_process_names:
            if process_name in self.config.processes and self.config.processes[process_name].enabled:
                fractionation_factors[process_name] = self.calculate_fractionation_factor(
                    process_name,
                    temperature,
                )
        return fractionation_factors
    
    def compute_bulk_isotope_fluxes(
        self,
        process_fluxes: Dict[str, torch.Tensor],
        source_ratios: Dict[str, torch.Tensor],
        temperature: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute 15N isotope fluxes for multiple processes.
        
        Parameters:
            process_fluxes: Total N fluxes by process
            source_ratios: 15N ratios by process
            temperature: Temperature [K]
            
        Returns:
            Dictionary mapping process names to 15N isotope fluxes
        """
        isotope_fluxes = {}
        for process_name in self.supported_process_names:
            if process_name in process_fluxes and process_name in source_ratios:
                if process_name in self.config.processes and self.config.processes[process_name].enabled:
                    isotope_fluxes[process_name] = self.calculate_isotope_flux(
                        process_name,
                        process_fluxes[process_name],
                        source_ratios[process_name],
                        temperature,
                    )
        return isotope_fluxes
    
    def get_isotope_info(self) -> Dict[str, Any]:
        """Get comprehensive 15N isotope information."""
        info = super().get_isotope_info()
        info.update({
            'standard_name': 'Air N2',
            'standard_ratio': self.air_n2_standard,
            'natural_abundance': self.air_n2_standard,
            'delta_range_typical': (-50.0, 100.0),  # Typical delta15N range in ?
            'process_details': {
                name: self.get_process_parameters(name)
                for name in self.supported_process_names
                if name in self.config.processes and self.config.processes[name].enabled
            }
        })
        return info
    
    def _compute_q10_fractionation(
        self,
        temperature: torch.Tensor,
        alpha_0: float,
        Q10: float,
        T_ref: float,
        process_name: str
    ) -> torch.Tensor:
        """
        Compute Q10 temperature-dependent fractionation.
        
        Mathematical Expression:
            alpha(T) = 1 + (alpha_0 - 1) * Q10^((T_ref - T)/10)
            
        References:
            - Mariotti et al. (1981). Experimental determination of nitrogen kinetic isotope fractionation
        """
        # References:
        #   - Mariotti et al. (1981). Experimental determination of nitrogen kinetic isotope fractionation.
        # Mathematical expression:
        #   alpha(T) = 1 + (alpha_0 - 1) * Q10^((T_ref - T)/10)
        # Parameter meanings:
        #   alpha_0: reference fractionation factor at T_ref [dimensionless]
        #   Q10: temperature coefficient [dimensionless]
        #   T_ref: reference temperature [K]
        #   temperature: process temperature [K]
        alpha = self.compute_q10_temperature_dependence(
            temperature=temperature,
            alpha_0=alpha_0,
            q10=Q10,
            t_ref=T_ref,
        )
        return alpha
    
    def _compute_arrhenius_fractionation(
        self,
        temperature: torch.Tensor,
        alpha_0: float,
        E_a: float,
        T_ref: float,
        process_name: str
    ) -> torch.Tensor:
        """
        Compute Arrhenius temperature-dependent fractionation.
        
        Mathematical Expression:
            alpha(T) = 1 + (alpha_0 - 1) * exp(Ea/R * (1/T - 1/T_ref))
            
        References:
            - Casciotti et al. (2003). Linking diversity and stable isotope fractionation
        """
        # References:
        #   - Casciotti et al. (2003). Linking diversity and stable isotope fractionation.
        # Mathematical expression:
        #   alpha(T) = 1 + (alpha_0 - 1) * exp((E_a / R) * (1 / T - 1 / T_ref))
        # Parameter meanings:
        #   alpha_0: reference fractionation factor at T_ref [dimensionless]
        #   E_a: activation energy [J mol-1]
        #   T_ref: reference temperature [K]
        #   R: gas constant [J mol-1 K-1]
        #   temperature: process temperature [K]
        alpha = self.compute_arrhenius_temperature_dependence(
            temperature=temperature,
            alpha_0=alpha_0,
            activation_energy=E_a,
            t_ref=T_ref,
        )
        return alpha
    
    def get_process_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get process summary for diagnostics.
        
        Returns:
            Dictionary with process configuration details
        """
        summary = {}
        for process_name in self.supported_process_names:
            # A process whose parameters cannot be read raises: the summary never
            # reports it as disabled.
            params = self.get_process_parameters(process_name)
            summary[process_name] = {
                'enabled': params['enabled'],
                'model_type': str(params['model_type']),
                'alpha_0': params['alpha_0'],
                'T_ref': params['T_ref'],
                'source_pool': params['source_pool'],
                'target_pool': params['target_pool'],
                'flux_variable': params['flux_variable']
            }

            # Add model-specific parameters
            if 'Q10' in params:
                summary[process_name]['Q10'] = params['Q10']
            if 'E_a' in params:
                summary[process_name]['E_a'] = params['E_a']
        
        return summary

    def __repr__(self) -> str:
        """String representation."""
        return f"Nitrogen15FractionationCalculator(processes={len(self.supported_process_names)})"


# Export the calculator
__all__ = ['Nitrogen15FractionationCalculator']

