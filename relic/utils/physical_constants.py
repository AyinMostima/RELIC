"""
Unified physical constants access interface.

This module provides a centralized access to physical constants,
mapping between different naming conventions and parameter database paths.

Design: Canonical snake_case keys with normalized lookup.
All lookups are normalized to snake_case before matching.

SSOT Policy:
  - Canonical data source: relic/parameters/database/relic/physical/constants.yaml
  - Access API: get_physical_constant(name)
  - Do not duplicate physical constant values in Python modules.
"""

from typing import Optional, Dict, Union, Iterable, Tuple
import re
import torch
from relic.parameters import get_global_manager


def _normalize_key(name: str) -> str:
    """
    Normalize a constant name to canonical snake_case.
    
    Handles:
      - SCREAMING_SNAKE_CASE -> snake_case
      - camelCase -> camel_case
      - PascalCase -> pascal_case
      - Mixed formats
    
    Examples:
        >>> _normalize_key('VON_KARMAN_CONSTANT')
        'von_karman_constant'
        >>> _normalize_key('gasConstant')
        'gas_constant'
        >>> _normalize_key('StefanBoltzmann')
        'stefan_boltzmann'
    """
    # Step 1: Insert underscore before uppercase letters (camelCase/PascalCase)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    # Step 2: Handle consecutive uppercase followed by lowercase (e.g., XMLParser -> xml_parser)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    # Step 3: Convert to lowercase
    return s.lower()


# Canonical mappings: use ONLY snake_case keys
# Aliases are handled via normalization, not duplication
_CANONICAL_MAPPINGS: Dict[str, str] = {
    # =========================================================================
    # FUNDAMENTAL PHYSICAL CONSTANTS (CODATA 2018)
    # =========================================================================
    'speed_of_light': 'relic.physical.constants.universal.speed_of_light',
    'c': 'relic.physical.constants.universal.speed_of_light',  # alias
    'planck_constant': 'relic.physical.constants.universal.planck_constant',
    'h': 'relic.physical.constants.universal.planck_constant',  # alias
    'reduced_planck_constant': 'relic.physical.constants.universal.reduced_planck_constant',
    'hbar': 'relic.physical.constants.universal.reduced_planck_constant',  # alias
    'boltzmann_constant': 'relic.physical.constants.universal.boltzmann_constant',
    'k_b': 'relic.physical.constants.universal.boltzmann_constant',  # alias
    'avogadro_constant': 'relic.physical.constants.universal.avogadro_constant',
    'avogadro_number': 'relic.physical.constants.universal.avogadro_constant',  # alias
    'n_a': 'relic.physical.constants.universal.avogadro_constant',  # alias
    'gas_constant': 'relic.physical.constants.universal.gas_constant',
    'r_gas': 'relic.physical.constants.universal.gas_constant',  # alias
    'r': 'relic.physical.constants.universal.gas_constant',  # alias (context-dependent)
    'stefan_boltzmann_constant': 'relic.physical.constants.universal.stefan_boltzmann_constant',
    'stefan_boltzmann': 'relic.physical.constants.universal.stefan_boltzmann_constant',  # alias
    'sigma': 'relic.physical.constants.universal.stefan_boltzmann_constant',  # alias
    'elementary_charge': 'relic.physical.constants.universal.elementary_charge',
    'e_charge': 'relic.physical.constants.universal.elementary_charge',  # alias
    'electron_mass': 'relic.physical.constants.universal.electron_mass',
    'm_e': 'relic.physical.constants.universal.electron_mass',  # alias
    'proton_mass': 'relic.physical.constants.universal.proton_mass',
    'm_p': 'relic.physical.constants.universal.proton_mass',  # alias
    'atomic_mass_unit': 'relic.physical.constants.universal.atomic_mass_unit',
    'amu': 'relic.physical.constants.universal.atomic_mass_unit',  # alias
    'u': 'relic.physical.constants.universal.atomic_mass_unit',  # alias
    'vacuum_permittivity': 'relic.physical.constants.universal.vacuum_permittivity',
    'epsilon_0': 'relic.physical.constants.universal.vacuum_permittivity',  # alias
    'vacuum_permeability': 'relic.physical.constants.universal.vacuum_permeability',
    'mu_0': 'relic.physical.constants.universal.vacuum_permeability',  # alias
    'gravitational_constant': 'relic.physical.constants.universal.gravitational_constant',
    'big_g': 'relic.physical.constants.universal.gravitational_constant',  # alias
    'fine_structure_constant': 'relic.physical.constants.universal.fine_structure_constant',
    'alpha': 'relic.physical.constants.universal.fine_structure_constant',  # alias
    'rydberg_constant': 'relic.physical.constants.universal.rydberg_constant',
    'r_inf': 'relic.physical.constants.universal.rydberg_constant',  # alias
    'faraday_constant': 'relic.physical.constants.universal.faraday_constant',
    'f': 'relic.physical.constants.universal.faraday_constant',  # alias
    'wien_displacement_constant': 'relic.physical.constants.universal.wien_displacement_constant',
    'b_wien': 'relic.physical.constants.universal.wien_displacement_constant',  # alias

    # =========================================================================
    # EARTH AND ATMOSPHERE CONSTANTS
    # =========================================================================
    'gravity_acceleration': 'relic.physical.constants.earth.gravity_acceleration',
    'gravity': 'relic.physical.constants.earth.gravity_acceleration',  # alias
    'g': 'relic.physical.constants.earth.gravity_acceleration',  # alias
    'earth_radius': 'relic.physical.constants.earth.radius_mean',
    'r_earth': 'relic.physical.constants.earth.radius_mean',  # alias
    'earth_angular_velocity': 'relic.physical.constants.earth.angular_velocity',
    'omega_earth': 'relic.physical.constants.earth.angular_velocity',  # alias
    'von_karman_constant': 'relic.physical.constants.turbulence.von_karman_constant',
    'kappa': 'relic.physical.constants.turbulence.von_karman_constant',  # alias
    'solar_constant': 'relic.physical.constants.solar.solar_constant',
    's_0': 'relic.physical.constants.solar.solar_constant',  # alias

    # =========================================================================
    # AIR PROPERTIES
    # =========================================================================
    'air_density_standard': 'relic.physical.constants.air.density_stp',
    'air_density_std': 'relic.physical.constants.air.density_stp',  # alias
    'air_density_stp': 'relic.physical.constants.air.density_stp',  # alias
    'air_specific_heat_cp': 'relic.physical.constants.air.specific_heat_cp',
    'specific_heat_air': 'relic.physical.constants.air.specific_heat_cp',  # alias
    'cp_air': 'relic.physical.constants.air.specific_heat_cp',  # alias
    'air_molecular_weight': 'relic.physical.constants.air.molecular_weight_dry',
    'air_mw': 'relic.physical.constants.air.molecular_weight_dry',  # alias
    'm_air': 'relic.physical.constants.air.molecular_weight_dry',  # alias
    'air_gamma': 'relic.physical.constants.air.gamma',
    'gamma_air': 'relic.physical.constants.air.gamma',  # alias
    'prandtl_number': 'relic.physical.constants.air.prandtl_number',
    'pr': 'relic.physical.constants.air.prandtl_number',  # alias

    # =========================================================================
    # WATER PROPERTIES
    # =========================================================================
    'water_density': 'relic.physical.constants.water.density_liquid',
    'rho_water': 'relic.physical.constants.water.density_liquid',  # alias
    'latent_heat_vaporization': 'relic.physical.constants.water.latent_heat_vaporization',
    'l_v': 'relic.physical.constants.water.latent_heat_vaporization',  # alias
    'water_molecular_weight': 'relic.physical.constants.water.molecular_weight',
    'water_mw': 'relic.physical.constants.water.molecular_weight',  # alias
    'm_water': 'relic.physical.constants.water.molecular_weight',  # alias
    'water_specific_heat': 'relic.physical.constants.water.specific_heat_liquid',
    'cp_water': 'relic.physical.constants.water.specific_heat_liquid',  # alias

    # =========================================================================
    # STANDARD CONDITIONS (TEMPERATURE/PRESSURE)
    # =========================================================================
    'standard_temperature': 'relic.physical.constants.temperature.standard_temperature',
    't0': 'relic.physical.constants.temperature.standard_temperature',  # alias (0 degC in K)
    'celsius_to_kelvin': 'relic.physical.constants.temperature.celsius_to_kelvin',
    't_freeze': 'relic.physical.constants.temperature.celsius_to_kelvin',  # alias
    'reference_temperature': 'relic.physical.constants.temperature.reference_temperature',
    't_ref': 'relic.physical.constants.parameters.t_ref',  # biosphere reference
    'absolute_zero': 'relic.physical.constants.temperature.absolute_zero',
    'standard_pressure': 'relic.physical.constants.pressure.standard_atmosphere',
    'p0': 'relic.physical.constants.pressure.standard_atmosphere',  # alias
    'atm': 'relic.physical.constants.pressure.standard_atmosphere',  # alias
    'reference_pressure': 'relic.physical.constants.pressure.reference_pressure',

    # =========================================================================
    # TIME CONVERSIONS
    # =========================================================================
    'seconds_per_minute': 'relic.physical.constants.time.seconds_per_minute',
    'seconds_per_hour': 'relic.physical.constants.time.seconds_per_hour',
    'seconds_per_day': 'relic.physical.constants.time.seconds_per_day',
    'seconds_per_year': 'relic.physical.constants.time.seconds_per_year',

    # =========================================================================
    # ATMOSPHERE COMPOSITION
    # =========================================================================
    'co2_reference': 'relic.physical.constants.atmosphere.co2_reference',
    'co2_current': 'relic.physical.constants.atmosphere.co2_current',
    'co2_preindustrial': 'relic.physical.constants.atmosphere.co2_preindustrial',
    'par_reference': 'relic.physical.constants.radiation.par_reference',
    'n2_fraction': 'relic.physical.constants.atmosphere.n2_fraction',
    'o2_fraction': 'relic.physical.constants.atmosphere.o2_fraction',

    # =========================================================================
    # MATHEMATICAL CONSTANTS
    # =========================================================================
    'pi': 'relic.physical.constants.math.pi',
    'two_pi': 'relic.physical.constants.math.two_pi',
    'e': 'relic.physical.constants.math.e',
    'euler_number': 'relic.physical.constants.math.e',  # alias
    'ln_2': 'relic.physical.constants.math.ln_2',
    'ln_10': 'relic.physical.constants.math.ln_10',
    'sqrt_2': 'relic.physical.constants.math.sqrt_2',
    'sqrt_3': 'relic.physical.constants.math.sqrt_3',
    'golden_ratio': 'relic.physical.constants.math.golden_ratio',
    'phi': 'relic.physical.constants.math.golden_ratio',  # alias
}


def _resolve_mapping(name: str) -> Union[str, Tuple[str, ...], None]:
    """
    Resolve a constant name to its parameter path(s).
    
    First tries exact match, then normalized match.
    Returns None if not found.
    """
    # Try exact match first (for performance)
    if name in _CANONICAL_MAPPINGS:
        return _CANONICAL_MAPPINGS[name]
    
    # Normalize and retry
    normalized = _normalize_key(name)
    if normalized in _CANONICAL_MAPPINGS:
        return _CANONICAL_MAPPINGS[normalized]
    
    return None


# Public API: CONSTANT_MAPPINGS exposes canonical mappings
CONSTANT_MAPPINGS: Dict[str, Union[str, Tuple[str, ...]]] = dict(_CANONICAL_MAPPINGS)


def get_physical_constant(
    name: str, 
    as_tensor: bool = False,
    device: Optional[torch.device] = None
) -> Union[float, torch.Tensor]:
    """
    Get physical constant from parameter manager with automatic mapping.
    
    Supports case-insensitive lookups:
      - 'von_karman_constant'
      - 'VON_KARMAN_CONSTANT'
      - 'vonKarmanConstant'
    
    Args:
        name: Common name or direct parameter path
        as_tensor: Return as torch.Tensor instead of float
        device: Device for tensor (if as_tensor=True)
        
    Returns:
        Physical constant value as float or tensor
        
    Examples:
        >>> get_physical_constant('von_karman_constant')
        0.41
        >>> get_physical_constant('relic.physical.constants.turbulence.von_karman_constant')
        0.41
    """
    param_manager = get_global_manager()
    
    # Try to get the mapped path or use direct path
    mapped = _resolve_mapping(name)
    if mapped is None:
        mapped = name
    if isinstance(mapped, (tuple, list)):
        param_paths: Iterable[str] = mapped
    else:
        param_paths = (mapped,)
    paths_display = ", ".join(str(p) for p in param_paths)
    
    # Try to get from parameter manager
    if param_manager:
        for param_path in param_paths:
            spec = param_manager.get_parameter(param_path)
            if spec is not None:
                if as_tensor:
                    return spec.to_tensor(device)
                value = param_manager.get(
                    param_path,
                    None,
                    units=spec.units,
                )
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        return float(value.detach().cpu().item())
                    return value
            else:
                value = param_manager.get(param_path, None)
                if value is not None:
                    if as_tensor:
                        return torch.as_tensor(value, device=device)
                    return value

    raise ValueError(
        f"Physical constant '{name}' not found in parameter database. "
        f"Tried path(s): {paths_display}"
    )


def get_all_physical_constants(
    as_dict: bool = True,
) -> Dict[str, float] | list[tuple[str, float]]:
    """
    Get all available physical constants.
    
    Args:
        as_dict: Return as dictionary (True) or list of tuples (False)
        
    Returns:
        Dictionary mapping constant names to values
    """
    get_global_manager()
    constants = {}
    
    # Get all mapped constants (using canonical keys only)
    for name, path in _CANONICAL_MAPPINGS.items():
        try:
            value = get_physical_constant(name)
            constants[name] = value
        except ValueError:
            pass
    
    if not as_dict:
        return list(constants.items())
    return constants


# Convenience functions for commonly used constants
def get_von_karman() -> float:
    """Get von Karman constant (~=0.41)."""
    return get_physical_constant('von_karman_constant')


def get_gravity() -> float:
    """Get gravitational acceleration (9.80665 m/s^2)."""
    return get_physical_constant('gravity_acceleration')


def get_gas_constant() -> float:
    """Get universal gas constant (8.314 J/mol/K)."""
    return get_physical_constant('gas_constant')


def get_stefan_boltzmann() -> float:
    """Get Stefan-Boltzmann constant (5.670374419e-08 W m-2 K-4)."""
    return get_physical_constant('stefan_boltzmann')


def get_standard_temperature() -> float:
    """Get standard temperature (273.15 K)."""
    return get_physical_constant('standard_temperature')


def get_standard_pressure() -> float:
    """Get standard pressure (101325 Pa)."""
    return get_physical_constant('standard_pressure')
