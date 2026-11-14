"""
Oxygen-18 Isotope Fractionation Calculator

This module implements 18O isotope fractionation calculations for water and
oxygen cycle processes including transpiration, evaporation, and water exchange.
The calculations follow the Craig-Gordon model and extensions.

Key Features:
- Craig-Gordon model for leaf water enrichment
- Equilibrium and kinetic fractionation factors
- Temperature-dependent fractionation
- Humidity and boundary layer effects
- Peclet effect for leaf water advection
- Integrated with PyTorch for automatic differentiation

Mathematical Framework:
    delta^1^8O = (Rsample/Rstandard - 1) * 1000?
    where Rstandard = ParameterManager('isotopes.standards.oxygen_18.vsmow_ratio') (VSMOW standard, validated)

Craig-Gordon Model:
    delta^1^8O_leaf = delta^1^8O_source + epsilon* + epsilonk * (ea - ei)/ea
    where epsilon* = equilibrium fractionation, epsilonk = kinetic fractionation

References:
    - Craig & Gordon (1965). Deuterium and oxygen-18 variations in the ocean
    - Flanagan et al. (1991). Leaf water 18O content and carbon isotope discrimination
    - Barnes & Allison (1988). The distribution of deuterium and 18O in dry soils
    - Yakir & Sternberg (2000). The use of stable isotopes to study ecosystem gas exchange
"""

import torch
from typing import Dict, Optional, Union
import warnings

from relic.parameters import UnifiedParameterManager

from ..config.multi_isotope_models import (
    OxygenFractionationParameters, OxygenProcessType
)
from .base import IsotopeFractionationCalculator
class Oxygen18FractionationCalculator(IsotopeFractionationCalculator):
    """
    Calculator for 18O isotope fractionation in water cycle processes.
    
    Implements process-specific fractionation models for oxygen isotope
    discrimination in transpiration, evaporation, and water exchange.
    
    Attributes:
        vsmow_standard: VSMOW standard ratio (0.0020052)
        supported_processes: Set of supported water/oxygen processes
        
    Methods:
        calculate_transpiration_fractionation: Craig-Gordon leaf water model
        calculate_evaporation_fractionation: Soil/canopy evaporation
        calculate_equilibrium_fractionation: Temperature-dependent equilibrium
        calculate_kinetic_fractionation: Diffusion kinetic effects
        craig_gordon_model: Complete Craig-Gordon leaf water enrichment
        peclet_correction: Peclet effect for leaf water advection
    """
    
    def __init__(
        self,
        *,
        parameter_manager: UnifiedParameterManager,
    ):
        """Initialize Oxygen-18 fractionation calculator."""
        super().__init__("18O", parameter_manager=parameter_manager)

        # VSMOW standard for 18O/16O ratio
        self.vsmow_standard = float(
            self.get_parameter_value(
                "vsmow_ratio",
                namespace="isotopes.standards.oxygen_18",
                required=True)
        )
        self.default_relative_humidity = float(
            self.get_parameter_value(
                "relative_humidity",
                namespace="isotopes.tracking",
                required=True)
        )
        self.horita_a = float(
            self.get_parameter_value(
                "equilibrium.horita_a",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.horita_b = float(
            self.get_parameter_value(
                "equilibrium.horita_b",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.horita_c = float(
            self.get_parameter_value(
                "equilibrium.horita_c",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.horita_d = float(
            self.get_parameter_value(
                "equilibrium.horita_d",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.kinetic_temp_exponent = float(
            self.get_parameter_value(
                "kinetic.temperature_exponent",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.wind_reference_speed = float(
            self.get_parameter_value(
                "boundary_layer.reference_wind_speed",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.boundary_layer_scaling = float(
            self.get_parameter_value(
                "boundary_layer.scaling_factor",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.stomatal_reference_conductance = float(
            self.get_parameter_value(
                "stomatal.reference_conductance",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.peclet_min = float(
            self.get_parameter_value(
                "peclet.min",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.peclet_max = float(
            self.get_parameter_value(
                "peclet.max",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.peclet_threshold = float(
            self.get_parameter_value(
                "peclet.threshold",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.peclet_taylor_denominator = float(
            self.get_parameter_value(
                "peclet.taylor_denominator",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.root_fractionation = float(
            self.get_parameter_value(
                "root_fractionation",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.root_temperature_sensitivity = float(
            self.get_parameter_value(
                "temperature_sensitivity",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.salinity_sensitivity = float(
            self.get_parameter_value(
                "salinity_sensitivity",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.relative_humidity_min = float(
            self.get_parameter_value(
                "relative_humidity.min",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.relative_humidity_max = float(
            self.get_parameter_value(
                "relative_humidity.max",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.temperature_warning_min = float(
            self.get_parameter_value(
                "warning_min",
                namespace="isotopes.temperature",
                required=True)
        )
        self.temperature_warning_max = float(
            self.get_parameter_value(
                "warning_max",
                namespace="isotopes.temperature",
                required=True)
        )
        self.discrimination_scale = float(
            self.get_parameter_value(
                "discrimination_scale",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.soil_kinetic_reduction = float(
            self.get_parameter_value(
                "evaporation.soil_kinetic_reduction",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.canopy_kinetic_reduction = float(
            self.get_parameter_value(
                "evaporation.canopy_kinetic_reduction",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.soil_moisture_reference = float(
            self.get_parameter_value(
                "evaporation.soil_moisture_reference",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.moisture_limitation_min = float(
            self.get_parameter_value(
                "evaporation.moisture_limitation_min",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )
        self.moisture_limitation_max = float(
            self.get_parameter_value(
                "evaporation.moisture_limitation_max",
                namespace="isotopes.fractionation.oxygen",
                required=True)
        )

        # Supported oxygen/water processes
        self.supported_processes = {
            OxygenProcessType.TRANSPIRATION,
            OxygenProcessType.SOIL_EVAPORATION,
            OxygenProcessType.CANOPY_EVAPORATION,
            OxygenProcessType.WATER_UPTAKE,
            OxygenProcessType.LEAF_WATER_EXCHANGE
        }
        
        
    def _build_default_oxygen_params(self) -> OxygenFractionationParameters:
        """Construct oxygen fractionation parameters from ParameterManager values."""
        return OxygenFractionationParameters(
            kinetic_fractionation=float(
                self.get_parameter_value(
                    "kinetic_fractionation",
                    namespace="isotopes.fractionation.oxygen",
                    required=True)
            ),
            reference_temperature=self.reference_temperature,
            humidity_sensitivity=float(
                self.get_parameter_value(
                    "humidity_sensitivity",
                    namespace="isotopes.fractionation.oxygen",
                    required=True)
            ),
            boundary_layer_conductance=float(
                self.get_parameter_value(
                    "boundary_layer_conductance",
                    namespace="isotopes.fractionation.oxygen",
                    required=True)
            ),
            peclet_number=float(
                self.get_parameter_value(
                    "peclet_number",
                    namespace="isotopes.fractionation.oxygen",
                    required=True)
            ),
        )

    # References: Craig and Gordon (1965); Yakir and Sternberg (2000).
    # Formula: alpha = f(process_type, T, humidity, boundary_conditions).
    # Parameters: process_type, temperature (K), oxygen_params (per mil coefficients).
    def calculate_fractionation_factor(
        self,
        process_type: Union[str, OxygenProcessType],
        temperature: torch.Tensor,
        oxygen_params: Optional[OxygenFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 18O fractionation factor for water cycle processes.
        
        Process-specific calculations:
        - Transpiration: Craig-Gordon model with leaf water enrichment
        - Evaporation: Modified Craig-Gordon for soil/canopy
        - Water uptake: Minimal fractionation (root discrimination)
        - Leaf water exchange: Equilibrium with atmospheric water vapor
        
        Parameters:
            process_type: Water/oxygen cycle process type
            temperature: Temperature [K], shape [..., 1]
            oxygen_params: Oxygen-specific fractionation parameters
            environmental_state: Environmental variables (humidity, VPD, etc.)
            **kwargs: Additional process-specific parameters
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
            
        Mathematical Expression:
            alpha = f(T, humidity, boundary_conditions) depending on process
            
        Raises:
            ValueError: For unsupported process types
            RuntimeError: For invalid parameter combinations
        """
        # Convert the declared canonical process name to its enum.
        if isinstance(process_type, str):
            try:
                process_type = OxygenProcessType(process_type.lower())
            except ValueError:
                raise ValueError(f"Unsupported oxygen process: {process_type}")
        
        # Validate process type
        if process_type not in self.supported_processes:
            raise ValueError(f"Process {process_type} not supported for 18O fractionation")
        
        # Ensure temperature is valid
        temperature = self._validate_temperature(temperature)
        
        # Process-specific fractionation calculations
        if process_type == OxygenProcessType.TRANSPIRATION:
            return self.calculate_transpiration_fractionation(
                temperature, oxygen_params, environmental_state, **kwargs
            )
            
        elif process_type in {
            OxygenProcessType.SOIL_EVAPORATION,
            OxygenProcessType.CANOPY_EVAPORATION
        }:
            return self.calculate_evaporation_fractionation(
                process_type, temperature, oxygen_params, environmental_state, **kwargs
            )
            
        elif process_type == OxygenProcessType.WATER_UPTAKE:
            return self.calculate_water_uptake_fractionation(
                temperature, oxygen_params, environmental_state, **kwargs
            )
            
        elif process_type == OxygenProcessType.LEAF_WATER_EXCHANGE:
            return self.calculate_leaf_water_exchange_fractionation(
                temperature, oxygen_params, environmental_state, **kwargs
            )
            
        else:
            raise ValueError(f"No fractionation model implemented for {process_type}")
    
    # References: Craig and Gordon (1965); Flanagan et al. (1991).
    # Formula: epsilon_total = epsilon_eq + epsilon_k * (1 - h); alpha = 1 / (1 + (epsilon_total * scale) / 1000).
    # Parameters: equilibrium_fractionation (per mil), kinetic_fractionation (per mil),
    #             relative_humidity (0-1), discrimination_scale (unitless).
    def calculate_transpiration_fractionation(
        self,
        temperature: torch.Tensor,
        oxygen_params: Optional[OxygenFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 18O fractionation for transpiration.
        
        Implements the Craig-Gordon model for leaf water enrichment:
        delta^1^8O_leaf = delta^1^8O_source + epsilon* + epsilonk * (ea - ei)/ea * (1 - h)
        
        With Peclet effect correction for leaf water advection:
        delta^1^8O_transpiration = delta^1^8O_leaf * (1 - e^(-P))/P
        
        Parameters:
            temperature: Temperature [K], shape [..., 1]
            oxygen_params: Oxygen fractionation parameters
            environmental_state: Environmental conditions (RH, VPD, wind, etc.)
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
            
        Mathematical Expression:
            epsilon_total = epsilon* + epsilonk * (1 - h) + Peclet_correction
        """
        # Set default parameters if not provided
        if oxygen_params is None:
            oxygen_params = self._build_default_oxygen_params()
        
        # Calculate equilibrium fractionation
        epsilon_eq = self.calculate_equilibrium_fractionation(temperature)
        
        # Calculate kinetic fractionation
        epsilon_kinetic = self.calculate_kinetic_fractionation(
            temperature, oxygen_params, environmental_state, **kwargs
        )
        
        # Extract environmental conditions
        relative_humidity = self._get_relative_humidity(
            environmental_state, reference_tensor=temperature, **kwargs
        )
        
        # Craig-Gordon model: epsilon = epsilon* + epsilonk * (1 - h)
        # where h is relative humidity (fractional)
        epsilon_total = epsilon_eq + epsilon_kinetic * (1.0 - relative_humidity)
        
        # Peclet effect correction
        epsilon_total = self.apply_peclet_correction(
            epsilon_total, temperature, oxygen_params, environmental_state, **kwargs
        )
        
        # Convert discrimination to fractionation factor (scaled to keep alpha near unity)
        epsilon_total = epsilon_total * self.discrimination_scale
        alpha = 1.0 / (1.0 + epsilon_total / 1000.0)
        
        return alpha
    
    # References: Craig and Gordon (1965); Barnes and Allison (1988).
    # Formula: epsilon_total = epsilon_eq + epsilon_k * (1 - h) * moisture_limitation; alpha = 1 / (1 + (epsilon_total * scale) / 1000).
    # Parameters: soil_kinetic_reduction, canopy_kinetic_reduction (unitless),
    #             soil_moisture_reference (unitless), moisture_limitation_min/max (unitless).
    def calculate_evaporation_fractionation(
        self,
        evaporation_type: OxygenProcessType,
        temperature: torch.Tensor,
        oxygen_params: Optional[OxygenFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 18O fractionation for evaporation processes.
        
        Evaporation fractionation is similar to transpiration but with:
        - Different boundary layer characteristics
        - Surface temperature effects
        - Soil moisture limitations
        - Reduced kinetic effects compared to leaves
        
        Parameters:
            evaporation_type: Type of evaporation (soil, canopy)
            temperature: Temperature [K], shape [..., 1] 
            oxygen_params: Oxygen fractionation parameters
            environmental_state: Environmental conditions
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
        """
        # Set default parameters if not provided
        if oxygen_params is None:
            oxygen_params = self._build_default_oxygen_params()
        
        # Calculate equilibrium fractionation
        epsilon_eq = self.calculate_equilibrium_fractionation(temperature)
        
        # Calculate kinetic fractionation (modified for surface evaporation)
        epsilon_kinetic = self.calculate_kinetic_fractionation(
            temperature, oxygen_params, environmental_state, **kwargs
        )
        
        # Modify kinetic fractionation based on evaporation type
        if evaporation_type == OxygenProcessType.SOIL_EVAPORATION:
            # Soil evaporation has reduced kinetic effects due to:
            # - Tortuosity of soil pore space
            # - Reduced turbulence near surface
            # - Soil moisture limitations
            kinetic_reduction = kwargs.get(
                'soil_kinetic_reduction', self.soil_kinetic_reduction
            )
            epsilon_kinetic = epsilon_kinetic * kinetic_reduction
            
        elif evaporation_type == OxygenProcessType.CANOPY_EVAPORATION:
            # Canopy evaporation (intercepted water) has different characteristics:
            # - Rapid evaporation (less time for enrichment)
            # - Different surface roughness
            kinetic_reduction = kwargs.get(
                'canopy_kinetic_reduction', self.canopy_kinetic_reduction
            )
            epsilon_kinetic = epsilon_kinetic * kinetic_reduction
        
        # Environmental conditions
        relative_humidity = self._get_relative_humidity(
            environmental_state, reference_tensor=temperature, **kwargs
        )
        
        # Surface moisture limitation (for soil evaporation)
        moisture_limitation = 1.0
        if (evaporation_type == OxygenProcessType.SOIL_EVAPORATION and
            environmental_state and 'soil_moisture' in environmental_state):
            soil_moisture = environmental_state['soil_moisture']
            # Reduced evaporation rate affects fractionation
            moisture_ref = kwargs.get(
                'soil_moisture_reference', self.soil_moisture_reference
            )
            min_limit = kwargs.get(
                'moisture_limitation_min', self.moisture_limitation_min
            )
            max_limit = kwargs.get(
                'moisture_limitation_max', self.moisture_limitation_max
            )
            moisture_limitation = torch.clamp(
                soil_moisture / moisture_ref, min=min_limit, max=max_limit
            )
        
        # Modified Craig-Gordon model for evaporation
        epsilon_total = epsilon_eq + epsilon_kinetic * (1.0 - relative_humidity) * moisture_limitation
        
        # Convert discrimination to fractionation factor (scaled to keep alpha near unity)
        epsilon_total = epsilon_total * self.discrimination_scale
        alpha = 1.0 / (1.0 + epsilon_total / 1000.0)
        
        return alpha
    
    # References: Horita and Wesolowski (1994), doi:10.1016/0016-7037(94)90096-5.
    # Formula: 1000 ln(alpha_l-v) = A + B/T + C/T^2 + D/T^3.
    # Parameters: equilibrium.horita_a/b/c/d with units 1, K, K2, and K3.
    def calculate_equilibrium_fractionation(
        self,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate temperature-dependent equilibrium fractionation.
        
        Uses the Horita and Wesolowski (1994) liquid-vapor regression over
        the declared isotope temperature domain.
        
        Parameters:
            temperature: Temperature [K], shape [..., 1]
        Returns:
            Equilibrium fractionation epsilon* [per mil]
        """
        temperature = self._validate_temperature(temperature)
        return (
            self.horita_a
            + self.horita_b / temperature
            + self.horita_c / temperature**2
            + self.horita_d / temperature**3
        )
    
    # References: Craig and Gordon (1965); Flanagan et al. (1991).
    # Formula: epsilon_k = epsilon_k_base * (T / T_ref)^n * f(gb, gs).
    # Parameters: kinetic_fractionation (per mil), kinetic.temperature_exponent (unitless),
    #             boundary_layer_conductance (m s-1), boundary_layer.reference_wind_speed (m s-1),
    #             boundary_layer.scaling_factor (unitless), stomatal.reference_conductance (mol m-2 s-1).
    def calculate_kinetic_fractionation(
        self,
        temperature: torch.Tensor,
        oxygen_params: Optional[OxygenFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate kinetic fractionation during diffusion.
        
        Kinetic fractionation depends on:
        - Molecular diffusivity differences
        - Boundary layer conductance
        - Turbulence and wind speed
        - Stomatal conductance (for transpiration)
        
        epsilonk = 32? (molecular diffusion in still air)
        Modified by boundary layer and stomatal effects
        
        Parameters:
            temperature: Temperature [K], shape [..., 1]
            oxygen_params: Oxygen fractionation parameters
            environmental_state: Environmental conditions
            
        Returns:
            Kinetic fractionation epsilonk [?]
        """
        if oxygen_params is None:
            oxygen_params = self._build_default_oxygen_params()
        
        # Base kinetic fractionation (molecular diffusion)
        epsilon_k_base = oxygen_params.kinetic_fractionation  # ~32?
        
        # References: kinetic theory for molecular diffusion.
        # Formula: D ~ T^n with n from kinetic.temperature_exponent.
        # Parameters: kinetic.temperature_exponent.
        T_ref = oxygen_params.reference_temperature
        temp_correction = (temperature / T_ref) ** self.kinetic_temp_exponent
        epsilon_k = epsilon_k_base * temp_correction
        
        # Boundary layer effects
        if (environmental_state and 'wind_speed' in environmental_state and
            oxygen_params.boundary_layer_conductance is not None):
            
            wind_speed = environmental_state['wind_speed']  # m/s
            gb = oxygen_params.boundary_layer_conductance  # Base conductance
            
            # Boundary layer conductance increases with wind speed
            # gb_effective = gb * (1 + wind_speed/wind_ref)
            wind_ref = kwargs.get('reference_wind_speed', self.wind_reference_speed)  # m/s
            gb_effective = gb * (1.0 + wind_speed / wind_ref)
            
            # Higher boundary layer conductance reduces kinetic fractionation
            conductance_effect = 1.0 / (1.0 + gb_effective * self.boundary_layer_scaling)
            epsilon_k = epsilon_k * conductance_effect
        
        # Stomatal conductance effects (for transpiration)
        if environmental_state and 'stomatal_conductance' in environmental_state:
            gs = environmental_state['stomatal_conductance']  # mol m-2 s-1
            
            # Higher stomatal conductance reduces kinetic fractionation
            # Following Flanagan et al. (1991) approach
            gs_ref = kwargs.get(
                'reference_stomatal_conductance',
                self.stomatal_reference_conductance,
            )  # mol m-2 s-1
            stomatal_effect = gs_ref / (gs + gs_ref)
            epsilon_k = epsilon_k * stomatal_effect
        
        return epsilon_k
    
    # References: Craig and Gordon (1965); Flanagan et al. (1991).
    # Formula: correction = (1 - exp(-P)) / P, with P = E * L / D.
    # Parameters: peclet_number and peclet.min/max/threshold/taylor_denominator [1].
    def apply_peclet_correction(
        self,
        epsilon_leaf: torch.Tensor,
        temperature: torch.Tensor,
        oxygen_params: Optional[OxygenFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply Peclet effect correction for leaf water advection.
        
        The Peclet effect accounts for the fact that transpired water
        represents a mixture of enriched leaf water and source water
        due to convective flow through the leaf.
        
        delta^1^8O_transpiration = delta^1^8O_leaf * (1 - e^(-P))/P
        
        where P is the Peclet number = E*L/D
        E = transpiration rate, L = leaf length, D = diffusivity
        
        Parameters:
            epsilon_leaf: Leaf water enrichment [?]
            temperature: Temperature [K]
            oxygen_params: Oxygen fractionation parameters
            environmental_state: Environmental conditions
            
        Returns:
            Corrected fractionation with Peclet effect [?]
        """
        if oxygen_params is None:
            oxygen_params = self._build_default_oxygen_params()
        peclet = oxygen_params.peclet_number
        
        # Ensure Peclet number is positive and reasonable
        if not isinstance(peclet, torch.Tensor):
            peclet = torch.tensor(peclet, dtype=temperature.dtype, device=temperature.device)
        peclet = torch.clamp(peclet, min=self.peclet_min, max=self.peclet_max)
        
        # Apply Peclet correction
        # For large P: correction ~= 1/P
        # For small P: correction ~= 1 - P/2
        peclet_correction = torch.where(
            peclet > self.peclet_threshold,
            (1.0 - torch.exp(-peclet)) / peclet,
            1.0 - peclet / self.peclet_taylor_denominator  # Taylor expansion for small P
        )
        
        epsilon_corrected = epsilon_leaf * peclet_correction
        
        return epsilon_corrected
    
    # References: Yakir and Sternberg (2000); Barnes and Allison (1988).
    # Formula: epsilon_total = base + temp_sensitivity * (T - T_ref) + salinity_sensitivity * salinity.
    # Parameters: root_fractionation (per mil), temperature_sensitivity (per mil K-1),
    #             salinity_sensitivity (per mil per salinity unit), temperature.reference (K).
    def calculate_water_uptake_fractionation(
        self,
        temperature: torch.Tensor,
        oxygen_params: Optional[OxygenFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 18O fractionation for water uptake by roots.
        
        Root water uptake generally shows minimal fractionation because:
        - Liquid-liquid equilibration is slow
        - No phase change involved
        - Bulk water movement rather than selective transport
        
        Small fractionation may occur due to:
        - Membrane permeability differences
        - Ion exchange processes
        - Metabolic discrimination
        
        Parameters:
            temperature: Temperature [K], shape [..., 1]
            oxygen_params: Oxygen fractionation parameters
            environmental_state: Environmental conditions
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
        """
        # Root water uptake shows minimal fractionation
        # Based on experimental observations: typically < 2?
        base_fractionation = kwargs.get('root_fractionation', self.root_fractionation)
        
        # Temperature dependence (very weak)
        T_ref = self.reference_temperature
        temp_sensitivity = kwargs.get('temperature_sensitivity', self.root_temperature_sensitivity)  # ?/K
        temp_correction = temp_sensitivity * (temperature - T_ref)
        
        # Soil salinity effects (increased fractionation with salinity)
        salinity_correction = 0.0
        if environmental_state and 'soil_salinity' in environmental_state:
            salinity = environmental_state['soil_salinity']  # dS/m
            # Higher salinity increases fractionation slightly
            salinity_correction = self.salinity_sensitivity * salinity
        
        # Total fractionation
        epsilon_total = base_fractionation + temp_correction + salinity_correction
        
        # Convert to fractionation factor
        alpha = 1.0 / (1.0 + epsilon_total / 1000.0)
        
        return alpha
    
    # References: Yakir and Sternberg (2000); Flanagan et al. (1991).
    # Formula: epsilon_exchange = epsilon_eq * (1 + humidity_sensitivity * h) * f(gs).
    # Parameters: humidity_sensitivity (unitless), stomatal.reference_conductance (mol m-2 s-1),
    #             relative_humidity (0-1), equilibrium_fractionation (per mil).
    def calculate_leaf_water_exchange_fractionation(
        self,
        temperature: torch.Tensor,
        oxygen_params: Optional[OxygenFractionationParameters] = None,
        environmental_state: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Calculate 18O fractionation for leaf water exchange with atmosphere.
        
        Leaf water can exchange with atmospheric water vapor through:
        - Stomatal exchange (bidirectional)
        - Cuticular water exchange
        - Equilibration processes
        
        This process tends to reduce leaf water enrichment by bringing
        leaf water closer to atmospheric water vapor composition.
        
        Parameters:
            temperature: Temperature [K], shape [..., 1]
            oxygen_params: Oxygen fractionation parameters
            environmental_state: Environmental conditions
            
        Returns:
            Fractionation factor alpha [dimensionless], shape [...]
        """
        if oxygen_params is None:
            oxygen_params = self._build_default_oxygen_params()
        
        # Equilibrium fractionation between liquid water and water vapor
        epsilon_eq = self.calculate_equilibrium_fractionation(temperature)
        
        # Exchange rate depends on:
        # - Stomatal conductance
        # - Leaf water content
        # - Atmospheric humidity
        
        relative_humidity = self._get_relative_humidity(
            environmental_state, reference_tensor=temperature, **kwargs
        )
        
        # Higher humidity promotes exchange (reduces driving force for enrichment)
        humidity_effect = kwargs.get('humidity_sensitivity', oxygen_params.humidity_sensitivity)
        humidity_correction = 1.0 + humidity_effect * relative_humidity
        
        # Stomatal conductance affects exchange rate
        conductance_effect = 1.0
        if environmental_state and 'stomatal_conductance' in environmental_state:
            gs = environmental_state['stomatal_conductance']
            gs_ref = kwargs.get(
                'reference_stomatal_conductance',
                self.stomatal_reference_conductance,
            )
            conductance_effect = (gs + gs_ref) / (2.0 * gs_ref)
        
        # Calculate exchange fractionation
        epsilon_exchange = epsilon_eq * humidity_correction * conductance_effect
        
        # Convert to fractionation factor
        alpha = 1.0 / (1.0 + epsilon_exchange / 1000.0)
        
        return alpha
    
    # References: Craig and Gordon (1965).
    # Formula: delta_leaf = delta_source + epsilon_eq + epsilon_k * (1 - h).
    # Parameters: equilibrium_fractionation (per mil), kinetic_fractionation (per mil),
    #             kinetic.temperature_exponent (unitless), relative_humidity (0-1).
    def craig_gordon_model(
        self,
        temperature: torch.Tensor,
        relative_humidity: torch.Tensor,
        source_delta: torch.Tensor,
        oxygen_params: Optional[OxygenFractionationParameters] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Complete Craig-Gordon model for leaf water enrichment.
        
        delta^1^8O_leaf = delta^1^8O_source + epsilon* + epsilonk * (ea - ei)/ea
                   = delta^1^8O_source + epsilon* + epsilonk * (1 - h)
        
        where:
        - epsilon*: equilibrium fractionation
        - epsilonk: kinetic fractionation
        - h: relative humidity (fractional)
        
        Parameters:
            temperature: Temperature [K]
            relative_humidity: Relative humidity [0-1]
            source_delta: Source water delta^1^8O [?]
            oxygen_params: Oxygen fractionation parameters
            
        Returns:
            Leaf water delta^1^8O [?]
        """
        if oxygen_params is None:
            oxygen_params = self._build_default_oxygen_params()
        
        # Calculate fractionation components
        epsilon_eq = self.calculate_equilibrium_fractionation(temperature)
        epsilon_kinetic = oxygen_params.kinetic_fractionation
        
        # Temperature correction for kinetic fractionation
        T_ref = oxygen_params.reference_temperature
        temp_correction = (temperature / T_ref) ** self.kinetic_temp_exponent
        epsilon_kinetic = epsilon_kinetic * temp_correction
        
        # Craig-Gordon equation
        delta_leaf = source_delta + epsilon_eq + epsilon_kinetic * (1.0 - relative_humidity)
        
        return delta_leaf
    
    # References: Coplen et al. (2006).
    # Formula: delta = (R_sample / R_standard - 1) * 1000.
    # Parameters: vsmow_standard (ratio), isotope_ratio (ratio).
    def delta_notation(self, isotope_ratio: torch.Tensor) -> torch.Tensor:
        """
        Convert isotope ratio to delta notation.
        
        delta^1^8O = (Rsample/RVSMOW - 1) * 1000?
        
        Parameters:
            isotope_ratio: 18O/16O ratio [dimensionless]
            
        Returns:
            Delta value delta^1^8O [?]
        """
        return (isotope_ratio / self.vsmow_standard - 1.0) * 1000.0
    
    # References: Coplen et al. (2006).
    # Formula: R_sample = R_standard * (delta / 1000 + 1).
    # Parameters: vsmow_standard (ratio), delta_value (per mil).
    def ratio_from_delta(self, delta_value: torch.Tensor) -> torch.Tensor:
        """
        Convert delta notation to isotope ratio.
        
        Rsample = RVSMOW * (delta^1^8O/1000 + 1)
        
        Parameters:
            delta_value: Delta value delta^1^8O [?]
            
        Returns:
            Isotope ratio 18O/16O [dimensionless]
        """
        return self.vsmow_standard * (delta_value / 1000.0 + 1.0)
    
    def _get_relative_humidity(
        self,
        environmental_state: Optional[Dict[str, torch.Tensor]],
        reference_tensor: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Extract relative humidity from environmental state or kwargs."""
        if environmental_state and 'relative_humidity' in environmental_state:
            rh = environmental_state['relative_humidity']
        elif 'relative_humidity' in kwargs:
            rh = kwargs['relative_humidity']
        else:
            # Default relative humidity
            rh = self.default_relative_humidity
        
        # Ensure it's a tensor and within bounds
        if not isinstance(rh, torch.Tensor):
            if reference_tensor is not None:
                rh = torch.tensor(
                    rh,
                    dtype=reference_tensor.dtype,
                    device=reference_tensor.device,
                )
            else:
                rh = torch.tensor(rh, dtype=torch.float32)
        
        # Clamp to reasonable bounds
        rh = torch.clamp(
            rh, min=self.relative_humidity_min, max=self.relative_humidity_max
        )
        
        return rh
    
    def _validate_temperature(self, temperature: torch.Tensor) -> torch.Tensor:
        """Validate temperature tensor and apply reasonable bounds."""
        if not isinstance(temperature, torch.Tensor):
            temperature = torch.tensor(temperature, dtype=torch.float32)
        
        # Check for reasonable temperature range (warnings only)
        if (
            torch.any(temperature < self.temperature_warning_min)
            or torch.any(temperature > self.temperature_warning_max)
        ):
            warnings.warn(
                "Temperature outside typical range. "
                "Fractionation calculations may be unreliable."
            )
        
        # Clamp to prevent numerical issues
        min_temp, max_temp = self.temperature_bounds
        temperature = torch.clamp(temperature, min=min_temp, max=max_temp)
        
        return temperature

# Export the calculator
__all__ = ['Oxygen18FractionationCalculator']

