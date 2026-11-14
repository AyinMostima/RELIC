"""
Internal Coupling Module

This module provides internal coupling mechanisms for the RELIC biogeochemical
model. It implements the coupling between different Earth system spheres within
the same modeling framework, focusing on process-level interactions.

The internal coupling module includes two active components:

1. Biosphere-Atmosphere Coupling (biosphere_atmosphere.py):
   - Computes leaf boundary-layer conductance from canopy wind speed
   - Converts velocity conductance to molar conductance through UnitConverter
   - Exposes no stomatal, evapotranspiration, or energy-balance duplicate path

2. Soil-Vegetation Coupling (soil_vegetation.py):
   - Implements Feddes root water uptake model
   - Couples soil water dynamics with transpiration demand
   - Models root distribution and water stress responses
   - Integrates with Richards equation or bucket soil models

Key Features:
- CF-1.8 compliant variable naming and units
- Water balance and dimensional transport checks
- Numerical stability for extreme environmental conditions
- Comprehensive scientific documentation with literature references
- Integration with PyTorch for automatic differentiation
- Modular design allowing flexible coupling configurations

Usage:
    from . import (
        BiosphereAtmosphereCoupling,
        SoilVegetationCoupling,
    )

from typing import Any, Optional
    
    # Initialize coupling functors
    bio_atm = BiosphereAtmosphereCoupling(name="biosphere_atmosphere", config=config)
    soil_veg = SoilVegetationCoupling(name="soil_vegetation", config=config)

Dependencies:
- Core framework: relic.core.functor, abstractions
- Utilities: relic.utils.physical_constants, unified_response_functions
- Scientific modules: atmosphere, biosphere, hydrosphere, pedosphere
- External libraries: torch, numpy

Mathematical Framework:
The internal coupling follows established biogeochemical principles:

- Leaf boundary conductance: g_b = C_v * sqrt(U_av / d_leaf)
- Feddes root uptake: S(z) = alpha(psi) * beta(z) * T_p

Conservation Laws:
- Boundary conductance is an instantaneous transport coefficient with no local
  mass or energy storage ledger.
- Soil-vegetation water balance is checked by its dedicated coupling functor.

References:
- CTSM 5.0 Technical Note, Eq. 2.5.127. Leaf boundary resistance
- Campbell, G.S. and Norman, J.M. (1998). Environmental Biophysics
- Feddes, R.A. et al. (1978). Simulation of field water use and crop yield
"""

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

from ...core.functor import FunctorBase
from .biosphere_atmosphere import BiosphereAtmosphereCoupling
from .soil_vegetation import SoilVegetationCoupling, RootDistributionParams, SoilWaterStressParams
from ...parameters.namespaces import ParameterNamespace, resolve_namespace
from .. import (
    SV_NAMESPACE,
)
from relic.parameters import get_global_manager

# Version information
__version__ = "1.0.0"
__author__ = "Yuchong Long"

# Export all coupling classes and parameter configurations
__all__ = [
    # Main coupling classes
    'BiosphereAtmosphereCoupling',
    'SoilVegetationCoupling', 
    
    # Parameter configuration classes
    'RootDistributionParams',
    'SoilWaterStressParams',
    
    # Utility functions
    'get_coupling_execution_order',
    'register_coupling_variables'
]


def _get_param_manager(parameter_manager: Optional[Any] = None) -> Optional[Any]:
    """
    Lazily resolve the parameter manager to avoid import-time initialization.
    """
    if parameter_manager is not None:
        return parameter_manager
    return get_global_manager()


def get_available_couplings():
    """
    Get list of available internal coupling functors.
    
    Returns:
        Dict[str, type]: Mapping of coupling names to functor classes
    """
    return {
        'biosphere_atmosphere': BiosphereAtmosphereCoupling,
        'soil_vegetation': SoilVegetationCoupling,
    }


def create_coupling_functor(
    coupling_type: str,
    name: str,
    config: dict,
    parameter_manager: Optional[Any] = None
) -> FunctorBase:
    """
    Factory function to create coupling functors.
    
    Parameters:
        coupling_type : str
            Type of coupling ('biosphere_atmosphere', 'soil_vegetation', 'water_carbon')
        name : str
            Name for the functor instance
        config : dict
            Configuration dictionary
            
    Returns:
        FunctorBase: Initialized coupling functor
        
    Raises:
        ValueError: If coupling_type is not recognized
    """
    available_couplings = get_available_couplings()
    
    if coupling_type not in available_couplings:
        raise ValueError(f"Unknown coupling type: {coupling_type}. "
                        f"Available types: {list(available_couplings.keys())}")
    
    coupling_class = available_couplings[coupling_type]
    return coupling_class(name=name, config=config, parameter_manager=parameter_manager)


def get_default_configs(parameter_manager: Optional[Any] = None):
    """
    Build default coupling configurations directly from the parameter database.

    No hard-coded constants are used. Missing parameters raise an explicit error so that
    the YAML registry remains the authoritative source of truth.
    """
    param_manager = _get_param_manager(parameter_manager)

    if param_manager is None:
        raise RuntimeError("Unable to access ParameterManager for coupling defaults.")

    def _require_value(key: str, *, namespace: Optional[str] = None, units: Optional[str] = None) -> Any:
        value = param_manager.get(
            key,
            namespace=namespace,
            units=units,
        )
        if value is None:
            ns_prefix = f"{namespace}." if namespace else ""
            raise KeyError(f"Required coupling parameter '{ns_prefix}{key}' is not registered.")
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return value

    def _require_float(key: str, *, namespace: Optional[str] = None, units: Optional[str] = None) -> float:
        raw_value = _require_value(key, namespace=namespace, units=units)
        try:
            return float(raw_value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Parameter '{key}' must be convertible to float.") from exc

    def _require_str(key: str, *, namespace: Optional[str] = None) -> str:
        raw_value = _require_value(key, namespace=namespace)
        return str(raw_value)

    def _require_sequence(key: str, *, namespace: Optional[str] = None, units: Optional[str] = None) -> List[float]:
        raw = _require_value(key, namespace=namespace, units=units)
        if isinstance(raw, (list, tuple)):
            return [float(item) for item in raw]
        raise TypeError(f"Parameter '{key}' must be a sequence, received {type(raw).__name__}")

    sv_ns = resolve_namespace(SV_NAMESPACE)

    # Scientific parameters are resolved by the Functor through ParameterManager.
    # The factory config therefore carries no duplicate stomatal or ET surface.
    biosphere_atmosphere_config = {}

    soil_vegetation_config = {
        'root_distribution': {
            'distribution_type': _require_str("distribution_type", namespace=sv_ns),
            'root_depth_max': _require_float("root_depth_max", namespace=sv_ns, units="m"),
            'shape_parameter': _require_float(
                "shape_parameter", namespace=sv_ns, units="m-1"
            ),
        },
        'water_stress': {
            'wilting_point_potential': _require_float("wilting_point_potential", namespace=sv_ns, units="Pa"),
            'anaerobic_point': _require_float(
                "anaerobic_point", namespace=sv_ns, units="Pa"
            ),
            'optimal_range': _require_sequence("optimal_range", namespace=sv_ns, units="Pa"),
        },
    }

    return {
        'biosphere_atmosphere': biosphere_atmosphere_config,
        'soil_vegetation': soil_vegetation_config,
    }


def get_coupling_execution_order():
    """
    Get recommended execution order for internal coupling functors.
    
    Returns:
        list: Recommended order of coupling types
    """
    return [
        'biosphere_atmosphere',  # Canopy wind -> boundary-layer conductance
        'soil_vegetation',       # ET demand -> root uptake -> soil moisture
    ]


# Module-level constants
COUPLING_TYPES = list(get_available_couplings().keys())
DEFAULT_EXECUTION_ORDER = get_coupling_execution_order()

def register_coupling_variables(validator, parameter_manager=None):
    """
    Register all output variables from internal coupling modules to CrossDomainValidator.
    
    This function registers the complete set of output variables produced by the
    internal coupling modules, ensuring proper unit tracking and validation.
    
    Parameters:
        validator : CrossDomainValidator
            The validator instance to register variables with
            
    Returns:
        int: Number of variables registered
    """
    from ...validation.cross_domain_validator import VariableSpec, DomainType

    pm = _get_param_manager(parameter_manager)
    if pm is None:
        raise RuntimeError("ParameterManager is required to load coupling variable bounds.")

    bounds_namespace_priority = (
        ParameterNamespace.COUPLING_VARIABLES,
        ParameterNamespace.COUPLING_CONSERVATION,
    )

    def _get_bounds(var_name: str, units: str):
        param_key = f"{var_name}.bounds"
        last_error: Optional[Exception] = None
        for ns in bounds_namespace_priority:
            resolved_ns = resolve_namespace(ns)
            try:
                bounds_value = pm.get(
                    param_key,
                    namespace=resolved_ns,
                    expected_type=list,
                    required=True,
                )
                if not isinstance(bounds_value, (list, tuple)) or len(bounds_value) != 2:
                    raise ValueError(
                        f"Bounds for {resolved_ns}.{param_key} must be a two-element list or tuple."
                    )
                return tuple(bounds_value)
            except KeyError as exc:
                last_error = exc
                continue
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
                raise RuntimeError(
                    f"Failed to load bounds for {resolved_ns}.{param_key}: {exc}"
                ) from exc
        searched = [resolve_namespace(ns) for ns in bounds_namespace_priority]
        raise KeyError(
            f"Bounds for {param_key} not found in namespaces {searched}; check parameter database definitions."
        ) from last_error
    
    registered_count = 0
    
    # Boundary-layer conductance is the sole retained output of the
    # biosphere-atmosphere transport closure. Stomatal, energy, and water-flux
    # outputs belong to their respective scientific process owners.
    ba_outputs = [
        VariableSpec("boundary_layer_conductance", "mol m-2 s-1", DomainType.ATMOSPHERE,
                    bounds=None,
                    description="Leaf boundary-layer gas conductance")
    ]
    
    # ===== Soil-Vegetation Coupling outputs =====
    sv_outputs = [
        VariableSpec("root_uptake_profile", "kg m-3 s-1", DomainType.HYDROSPHERE,
                    bounds=None,
                    description="Root water uptake rate by soil layer"),
        VariableSpec("effective_transpiration", "kg m-2 s-1", DomainType.HYDROSPHERE,
                    bounds=None,
                    description="Actual transpiration achieved after water stress"),
        VariableSpec("diagnostics.soil_vegetation.feddes_stress_factor", "dimensionless", DomainType.BIOSPHERE,
                    bounds=None,
                    description="Root-weighted Feddes water stress factor"),
        VariableSpec("diagnostics.soil_vegetation.root_zone_moisture", "m3 m-3", DomainType.HYDROSPHERE,
                    bounds=None,
                    description="Root-zone weighted average soil moisture"),
        VariableSpec("diagnostics.soil_vegetation.root_distribution", "m-1", DomainType.BIOSPHERE,
                    bounds=None,
                    description="Normalized root distribution profile"),
        VariableSpec("diagnostics.soil_vegetation.water_balance_residual", "kg m-2 s-1", DomainType.HYDROSPHERE,
                    bounds=None,
                    description="Depth-integrated root uptake minus effective transpiration"),
    ]
    
    # Load bounds from parameter system before registration
    for var in ba_outputs:
        var.bounds = _get_bounds(var.name, var.units)
    for var in sv_outputs:
        var.bounds = _get_bounds(var.name, var.units)

    # Register all variables
    for var in ba_outputs:
        validator.register_variable(var)
        registered_count += 1
        
    for var in sv_outputs:
        validator.register_variable(var)
        registered_count += 1
        
    return registered_count




