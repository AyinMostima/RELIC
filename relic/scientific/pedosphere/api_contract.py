"""
API Contract and Output Specification for Pedosphere Module

This module defines the standardized input/output contracts for all pedosphere
functors, ensuring consistent interfaces for integration with the RELIC framework.

All pedosphere functors must conform to these specifications for proper
integration with StateManager and other modules.
"""

from typing import Any, Dict, List, Protocol, Optional, Tuple
from enum import Enum
import torch
from dataclasses import dataclass
from relic.core.state import UpdateItem
from relic.parameters import get_global_manager

# Parameter management imports
from relic.core.functor import ProcessCategory, ExecutionPhase


def _resolve_manager(parameter_manager: Optional[object] = None) -> object:
    """Return explicitly injected parameter manager for advanced lookups."""
    if parameter_manager is not None:
        return parameter_manager
    return get_global_manager()


def _scalar_from_value(value: Any) -> float:
    """Convert numeric values to Python float without tensor.item/cpu APIs."""
    if isinstance(value, torch.Tensor):
        detached = value.detach().reshape(-1)
        if detached.numel() == 0:
            raise ValueError("Cannot convert empty tensor to scalar.")
        return detached[:1].to(dtype=torch.float64).tolist()[0]
    return float(value)

CONTRACT_METADATA = {
    "name": "pedosphere_contract",
    "version": "1.0.0",
    "description": "Pedosphere module contract metadata for standardized inputs/outputs.",
    "category": ProcessCategory.BIOLOGICAL_PROCESS,
    "execution_phase": ExecutionPhase.MAIN_PROCESSES,
    "references": [
        "CF-1.8 Convention (CF Conventions)",
        "FAIR Guiding Principles (Wilkinson et al., 2016)",
    ],
    "mathematical_expression": (
        "dM/dt = sum(inputs) - sum(outputs) + sum(internal transfers)"
    ),
    "conservation_laws": ["mass"],
    "parameters": {
        "flux_units": (
            "Flux units follow g C m-2 s-1 for carbon and g N m-2 s-1 for nitrogen."
        ),
        "pool_units": (
            "Pool units follow g C m-2 for carbon and g N m-2 for nitrogen."
        ),
    },
    "input_variables": [],
    "output_variables": [],
    "spatial_dimensions": 0,
    "temporal_scale": "second",
    "coupling_interfaces": [],
}


class FluxDirection(Enum):
    """Direction of material flux."""
    INPUT = "input"      # Into the system
    OUTPUT = "output"    # Out of the system
    INTERNAL = "internal"  # Between pools


class FluxType(Enum):
    """Type of flux for categorization."""
    RESPIRATION = "respiration"
    LEACHING = "leaching"
    TRANSFORMATION = "transformation"
    EMISSION = "emission"
    UPTAKE = "uptake"
    MASS = "mass"  # General mass flux


class PoolType(Enum):
    """Type of pool for categorization."""
    CARBON = "carbon"
    NITROGEN = "nitrogen"
    PHOSPHORUS = "phosphorus"
    WATER = "water"
    ENERGY = "energy"
    MICROBIAL = "microbial"


@dataclass
class FluxSpecification:
    """
    Specification for a flux variable.
    
    Attributes:
    -----------
    name : str
        Variable name in output dictionary
    units : str
        Physical units (must be g m-2 s-1 for mass fluxes)
    direction : FluxDirection
        Direction of flux
    flux_type : FluxType
        Category of flux
    description : str
        Scientific description
    is_rate : bool
        True if this is a rate (per second), False if total change
    """
    name: str
    units: str
    direction: FluxDirection
    flux_type: FluxType
    description: str
    is_rate: bool = True


@dataclass
class PoolSpecification:
    """
    Specification for a pool variable.
    
    Attributes:
    -----------
    name : str
        Variable name in state dictionary
    units : str
        Physical units (must be g m-2 for pools)
    pool_type : PoolType
        Category of pool
    description : str
        Scientific description
    min_value : Optional[float]
        Minimum allowed value (None if not specified)
    max_value : Optional[float]
        Maximum allowed value (None if not specified)
    """
    name: str
    units: str
    pool_type: PoolType
    description: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class PedosphereFunctorContract(Protocol):
    """
    Protocol defining the required interface for all pedosphere functors.
    
    All pedosphere functors must implement this interface for proper
    integration with the RELIC framework.
    Use PedosphereContractMixin to provide default implementations.
    """
    
    def get_required_inputs(self) -> List[str]:
        """
        Get list of required input variables from StateManager.

        FunctorBase provides it: the declared input variables minus
        ``optional_input_variables``.
        
        Returns:
        --------
        List[str]
            Required state variable names
        """
        ...
    
    def get_output_specifications(self) -> Dict[str, FluxSpecification]:
        """
        Get specifications for all output variables.
        
        Returns:
        --------
        Dict[str, FluxSpecification]
            Output variable specifications
        """
        ...
    
    def validate_inputs(self, state: Dict[str, torch.Tensor]) -> List[str]:
        """
        Validate input state variables.
        
        Parameters:
        -----------
        state : Dict[str, torch.Tensor]
            Input state from StateManager
            
        Returns:
        --------
        List[str]
            List of validation errors (empty if valid)
        """
        ...
    
    def forward(self,
        # References: RELIC architecture overview, Functor lifecycle contract;
        #   Manzoni and Porporato (2009), pedosphere flux accounting.
        # Mathematical expression: y = F(x, t, dt), where y is the FluxDict declared by the
        #   implementing Functor; this Protocol method performs no calculation.
        # Parameter meanings: x is the StateManager state, t the simulation time, dt the step,
        #   y the outputs declared by get_output_specifications.
        # Units: state and outputs carry the units declared by the implementing Functor's
        #   specifications; t and dt [s].
        # Domain/constraints: structural interface only; implementations validate inputs, keep
        #   one governed forward and return UpdateItems in the declared units.
        # Sign convention: defined by each implementing process.
               state: Dict[str, torch.Tensor],
               time: float,
               dt: torch.Tensor | float,
               **kwargs) -> Dict[str, torch.Tensor]:
        """
        Execute the functor for one time step.
        
        Parameters:
        -----------
        state : Dict[str, torch.Tensor]
            Current state from StateManager
        time : float
            Current simulation time [s]
        dt : torch.Tensor | float
            Time step size [s]
        **kwargs
            Additional parameters
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Output fluxes and state changes
        """
        ...


class PedosphereContractMixin:
    """
    Default implementations for pedosphere contract methods.

    Uses FunctorMetadata input variables and optional output_specifications when present.
    """

    def step_ledger_residual(
        self,
        output_updates: Dict[str, Any],
        *,
        increments: Tuple[str, ...],
        rates: Tuple[str, ...],
        step_key: str,
        increment_unit: str,
        rate_unit: str,
    ) -> torch.Tensor:
        """Recompute a closed per-step element ledger from a returned proposal.

        References:
          - Manzoni and Porporato (2009), Soil Biology and Biochemistry 41:1355-1379.
          - RELIC architecture overview, Section 20, local conservation hook contract.
        Mathematical expression:
          r = sum_i Delta_i + (sum_j F_j) * dt_step.
        Parameter meanings:
          Delta_i are the per-step pool increments returned as UpdateItem(per_timestep=True);
          F_j are the exported rates returned as UpdateItem(per_timestep=False); dt_step is
          the step length the proposal declares under step_key. Raw tensors (the
          forward-internal ledger of per-step deltas) are summed directly.
        Units:
          Delta_i and r [increment_unit]; F_j [rate_unit]; dt_step [s].
        Domain/constraints:
          every listed key must be present; UpdateItems must carry the declared unit and
          per_timestep flag; dt_step must be one finite positive scalar. Violations raise.
        Sign convention:
          pool increments are positive when the pool grows and exports are positive out of
          the pool, so a closed ledger gives r = 0.
        """
        keys = (*increments, *rates)
        missing = [key for key in keys if key not in output_updates]
        if missing:
            raise ValueError(f"{self.name}: conservation ledger requires outputs {missing}")
        values = {key: output_updates[key] for key in keys}
        if not any(isinstance(value, UpdateItem) for value in values.values()):
            residual = None
            for value in values.values():
                tensor = torch.as_tensor(value)
                residual = tensor if residual is None else residual + tensor
            return residual
        if step_key not in output_updates:
            raise ValueError(f"{self.name}: conservation ledger requires the step length '{step_key}'")
        step = output_updates[step_key]
        if isinstance(step, UpdateItem):
            step = step.value
        step = torch.as_tensor(step)
        if step.numel() != 1 or not bool(torch.isfinite(step).all()) or bool(step.reshape(()) <= 0.0):
            raise ValueError(f"{self.name}: '{step_key}' must be one finite positive scalar")

        def _typed_value(key: str, unit: str, per_timestep: bool) -> torch.Tensor:
            item = values[key]
            if not isinstance(item, UpdateItem):
                raise TypeError(f"{self.name}: output '{key}' must be an UpdateItem in the public ledger")
            if item.unit != unit or bool(item.per_timestep) != per_timestep:
                raise ValueError(
                    f"{self.name}: output '{key}' must carry unit '{unit}' with per_timestep={per_timestep}"
                )
            return torch.as_tensor(item.value)

        residual = None
        for key in increments:
            tensor = _typed_value(key, increment_unit, True)
            residual = tensor if residual is None else residual + tensor
        exported = None
        for key in rates:
            tensor = _typed_value(key, rate_unit, False)
            exported = tensor if exported is None else exported + tensor
        step = step.reshape(()).to(device=exported.device, dtype=exported.dtype)
        return residual + exported * step

    @staticmethod
    def _bool_from_config_value(value: Any) -> bool:
        """Normalize scalar config or parameter values to bool."""

        if isinstance(value, torch.Tensor):
            flat = value.detach().reshape(-1)
            if flat.numel() == 0:
                return False
            return bool(flat[:1].to(dtype=torch.float64).tolist()[0])
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on", "strict"}
        return bool(value)

    def _iphs_reaction_conservation_kwargs(
        self,
        *,
        conservation_matrix: Optional[torch.Tensor] = None,
        open_system_route: Optional[str] = None,
        default_strict: bool = False,
    ) -> Dict[str, Any]:
        """Return configurable IPHS reaction conservation options.

        References:
        - RELIC Phase 8 IPHS-DEC runtime gate.
        - Feinberg (2019), chemical reaction network stoichiometry.

        Mathematical expression:
        - Closed ledgers validate C_h N_h = 0.
        - Open systems declare a route instead of silently omitting C_h.

        Parameter meanings:
        - validation.conservation.strict_conservation toggles blocking enforcement.
        - phase8_strict_reaction_conservation is a module-level override.
        - open_system_route names an explicit boundary, port, or diagnostic route.
        """

        config = getattr(self, "config", None)
        if not isinstance(config, dict):
            config = getattr(self, "_config_raw", {})
        if not isinstance(config, dict):
            config = {}

        strict_value = config.get(
            "phase8_strict_reaction_conservation",
            config.get(
                "require_reaction_conservation_matrix",
                config.get("strict_reaction_conservation"),
            ),
        )
        if strict_value is None:
            getter = getattr(self, "get_parameter_value", None)
            if callable(getter):
                strict_value = getter(
                    "strict_conservation",
                    namespace="validation.conservation",
                    required=False,
                )
        strict = (
            self._bool_from_config_value(strict_value)
            if strict_value is not None
            else bool(default_strict)
        )

        options: Dict[str, Any] = {
            "strict_conservation": strict,
            "require_conservation_matrix": strict,
        }
        if conservation_matrix is not None:
            options["conservation_matrix"] = conservation_matrix
        if open_system_route is not None:
            options["open_system_route"] = open_system_route
        return options

    def get_output_specifications(self) -> Dict[str, FluxSpecification]:
        """Return output specifications if defined, else an empty mapping."""
        output_specifications = getattr(self, "output_specifications", None)
        if isinstance(output_specifications, dict):
            return output_specifications
        return {}

    def validate_inputs(self, state: Dict[str, torch.Tensor]) -> List[str]:
        """Validate inputs using FunctorBase.validate_state when available."""
        return self.validate_state(state)


# Standard output specifications for pedosphere functors
class StandardPedosphereOutputs:
    """
    Standard output specifications for pedosphere processes.
    
    These define the canonical output variables that should be used
    consistently across all pedosphere functors.
    """
    
    # Carbon fluxes
    HETEROTROPHIC_RESPIRATION = FluxSpecification(
        name="heterotrophic_respiration",
        units="g C m-2 s-1",
        direction=FluxDirection.OUTPUT,
        flux_type=FluxType.RESPIRATION,
        description="CO2 flux from heterotrophic respiration",
        is_rate=True
    )
    
    CARBON_LEACHING = FluxSpecification(
        name="carbon_leaching_flux",
        units="g C m-2 s-1",
        direction=FluxDirection.OUTPUT,
        flux_type=FluxType.LEACHING,
        description="Dissolved organic carbon leaching",
        is_rate=True
    )
    
    # Nitrogen fluxes
    MINERALIZATION = FluxSpecification(
        name="mineralization_flux",
        units="g N m-2 s-1",
        direction=FluxDirection.INTERNAL,
        flux_type=FluxType.TRANSFORMATION,
        description="Net nitrogen mineralization (positive) or immobilization (negative)",
        is_rate=True
    )
    
    NITRIFICATION = FluxSpecification(
        name="nitrification_flux",
        units="g N m-2 s-1",
        direction=FluxDirection.INTERNAL,
        flux_type=FluxType.TRANSFORMATION,
        description="NH4 to NO3 transformation",
        is_rate=True
    )
    
    DENITRIFICATION = FluxSpecification(
        name="denitrification_flux",
        units="g N m-2 s-1",
        direction=FluxDirection.OUTPUT,
        flux_type=FluxType.EMISSION,
        description="NO3 loss via denitrification",
        is_rate=True
    )
    
    N2O_EMISSION = FluxSpecification(
        name="n2o_flux",
        units="g N m-2 s-1",
        direction=FluxDirection.OUTPUT,
        flux_type=FluxType.EMISSION,
        description="N2O emission to atmosphere",
        is_rate=True
    )
    
    NITROGEN_LEACHING = FluxSpecification(
        name="nitrogen_leaching_flux",
        units="g N m-2 s-1",
        direction=FluxDirection.OUTPUT,
        flux_type=FluxType.LEACHING,
        description="Total nitrogen leaching (NO3 + DON)",
        is_rate=True
    )
    
    # Pool changes (deltas)
    DELTA_FAST_C = FluxSpecification(
        name="delta_fast_som_c",
        units="g C m-2 s-1",
        direction=FluxDirection.INTERNAL,
        flux_type=FluxType.TRANSFORMATION,
        description="Rate of change in fast SOM carbon pool",
        is_rate=True
    )
    
    DELTA_SLOW_C = FluxSpecification(
        name="delta_slow_som_c",
        units="g C m-2 s-1",
        direction=FluxDirection.INTERNAL,
        flux_type=FluxType.TRANSFORMATION,
        description="Rate of change in slow SOM carbon pool",
        is_rate=True
    )
    
    DELTA_PASSIVE_C = FluxSpecification(
        name="delta_passive_som_c",
        units="g C m-2 s-1",
        direction=FluxDirection.INTERNAL,
        flux_type=FluxType.TRANSFORMATION,
        description="Rate of change in passive SOM carbon pool",
        is_rate=True
    )


def _get_parameter_bounds(
    key: str,
    parameter_manager: Optional[object] = None,
) -> Tuple[Optional[float], Optional[float]]:
    manager = _resolve_manager(parameter_manager)
    if manager is None:
        return None, None
    spec = manager.get_parameter(key)
    if spec is None:
        return None, None
    # A spec without bounds is unbounded (None, None); a malformed bounds entry raises.
    bounds = getattr(spec, "bounds", None)
    if not bounds:
        return None, None
    min_val, max_val = bounds
    min_val = float(min_val) if min_val is not None else None
    max_val = float(max_val) if max_val is not None else None
    return min_val, max_val


def _get_parameter_value(
    key: str,
    parameter_manager: Optional[object] = None,
) -> Optional[float]:
    manager = _resolve_manager(parameter_manager)
    if manager is None:
        return None
    # An absent key is None (required=False); any manager error propagates.
    value = manager.get(
        key,
        required=False,
        expected_type=None,
    )
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _scalar_from_value(value)
    return _scalar_from_value(value)


# Standard pool specifications
_CARBON_POOL_ALIASES = {
    "doc",
    "dissolved_organic_carbon",
    "fast_carbon",
    "slow_carbon",
    "passive_carbon",
    "microbial_carbon",
    "enzyme_carbon",
}
_SOM_POOL_ALIASES = {"fast_carbon", "slow_carbon", "passive_carbon"}


def _collect_parameter_pool_names(
    namespace: str,
    suffix: str,
    parameter_manager: Optional[object] = None,
) -> List[str]:
    manager = _resolve_manager(parameter_manager)
    if manager is None:
        return []
    specs = manager.get_namespace_parameters(namespace)
    pool_names = []
    for key in specs.keys():
        if key.endswith(suffix):
            pool_names.append(key.split(".")[-1])
    return pool_names


def get_pedosphere_carbon_pool_names(
    pools: Optional[Dict[str, torch.Tensor]] = None,
    parameter_manager: Optional[object] = None,
) -> List[str]:
    names = set(
        _collect_parameter_pool_names(
            "pedosphere.carbon_pools",
            "_c",
            parameter_manager=parameter_manager,
        )
    )
    if pools:
        for key in pools.keys():
            if key.endswith("_c") or key in _CARBON_POOL_ALIASES:
                names.add(key)
    return sorted(names)


def get_pedosphere_carbon_pool_categories(
    pools: Optional[Dict[str, torch.Tensor]] = None,
    parameter_manager: Optional[object] = None,
) -> Dict[str, List[str]]:
    pool_names = get_pedosphere_carbon_pool_names(
        pools,
        parameter_manager=parameter_manager,
    )
    categories: Dict[str, List[str]] = {
        "litter": [],
        "som": [],
        "microbial": [],
        "dissolved": [],
        "other": [],
    }
    for name in pool_names:
        lowered = name.lower()
        if "litter" in lowered:
            categories["litter"].append(name)
        elif "som" in lowered or name in _SOM_POOL_ALIASES:
            categories["som"].append(name)
        elif "microbial" in lowered or "enzyme" in lowered:
            categories["microbial"].append(name)
        elif "doc" in lowered or "dissolved" in lowered:
            categories["dissolved"].append(name)
        else:
            categories["other"].append(name)
    for value in categories.values():
        value.sort()
    return categories


class _LazyPoolMeta(type):
    def __getattr__(cls, name: str) -> PoolSpecification:
        if name in cls._POOL_DEFINITIONS:
            return cls._build_pool_spec(name)
        raise AttributeError(f"{cls.__name__} has no attribute {name}")


class StandardPedospherePools(metaclass=_LazyPoolMeta):
    """
    Standard pool specifications for pedosphere state variables.
    """

    _POOL_DEFINITIONS = {
        "FAST_CARBON": {
            "name": "fast_carbon",
            "units": "g C m-2",
            "pool_type": PoolType.CARBON,
            "description": "Fast decomposition carbon pool",
            "bounds_key": "pedosphere.carbon_pools.fast_som_c",
        },
        "SLOW_CARBON": {
            "name": "slow_carbon",
            "units": "g C m-2",
            "pool_type": PoolType.CARBON,
            "description": "Slow decomposition carbon pool",
            "bounds_key": "pedosphere.carbon_pools.slow_som_c",
        },
        "PASSIVE_CARBON": {
            "name": "passive_carbon",
            "units": "g C m-2",
            "pool_type": PoolType.CARBON,
            "description": "Passive or recalcitrant carbon pool",
            "bounds_key": "pedosphere.carbon_pools.passive_som_c",
        },
        "MICROBIAL_CARBON": {
            "name": "microbial_carbon",
            "units": "g C m-2",
            "pool_type": PoolType.CARBON,
            "description": "Microbial biomass carbon",
            # Bounds of the owner process (microbial-enzyme decomposition), not an
            # environment default.
            "min_key": "pedosphere.decomposition.microbial_enzyme.microbial.min_biomass",
            "max_key": "pedosphere.decomposition.microbial_enzyme.microbial.max_biomass",
        },
        "ORGANIC_N": {
            "name": "orgN",
            "units": "g N m-2",
            "pool_type": PoolType.NITROGEN,
            "description": "Organic nitrogen pool",
        },
        "NH4": {
            "name": "NH4",
            "units": "g N m-2",
            "pool_type": PoolType.NITROGEN,
            "description": "Ammonium nitrogen pool",
            "min_key": "pedosphere.numerical.min_concentration_nh4",
            "max_key": "pedosphere.numerical.max_concentration_nh4",
        },
        "NO3": {
            "name": "NO3",
            "units": "g N m-2",
            "pool_type": PoolType.NITROGEN,
            "description": "Nitrate nitrogen pool",
            "min_key": "pedosphere.numerical.min_concentration_no3",
            "max_key": "pedosphere.numerical.max_concentration_no3",
        },
    }

    @classmethod
    def _build_pool_spec(
        cls,
        key: str,
        parameter_manager: Optional[object] = None,
    ) -> PoolSpecification:
        definition = cls._POOL_DEFINITIONS[key]
        min_value = None
        max_value = None
        bounds_key = definition.get("bounds_key")
        if bounds_key:
            min_value, max_value = _get_parameter_bounds(
                bounds_key,
                parameter_manager=parameter_manager,
            )
        min_key = definition.get("min_key")
        max_key = definition.get("max_key")
        if min_key:
            min_value = _get_parameter_value(min_key, parameter_manager=parameter_manager)
        if max_key:
            max_value = _get_parameter_value(max_key, parameter_manager=parameter_manager)
        return PoolSpecification(
            name=definition["name"],
            units=definition["units"],
            pool_type=definition["pool_type"],
            min_value=min_value,
            max_value=max_value,
            description=definition["description"],
        )

    @classmethod
    def get_pool_spec(
        cls,
        key: str,
        parameter_manager: Optional[object] = None,
    ) -> PoolSpecification:
        return cls._build_pool_spec(key, parameter_manager=parameter_manager)

    @classmethod
    def list_pool_specs(
        cls,
        parameter_manager: Optional[object] = None,
    ) -> Dict[str, PoolSpecification]:
        return {
            key: cls._build_pool_spec(key, parameter_manager=parameter_manager)
            for key in cls._POOL_DEFINITIONS
        }


def validate_functor_output(output: Dict[str, torch.Tensor],
                           specifications: Dict[str, FluxSpecification]) -> List[str]:
    """
    Validate functor output against specifications.
    
    Parameters:
    -----------
    output : Dict[str, torch.Tensor]
        Functor output dictionary
    specifications : Dict[str, FluxSpecification]
        Expected output specifications
        
    Returns:
    --------
    List[str]
        Validation errors (empty if valid)
    """
    errors = []
    
    # Check for missing outputs
    for spec_name, spec in specifications.items():
        if spec.name not in output:
            errors.append(f"Missing required output: {spec.name}")
    
    # Check for unexpected outputs
    expected_names = {spec.name for spec in specifications.values()}
    for output_name in output.keys():
        if output_name not in expected_names:
            # Warning only, not an error
            pass  # Could log this as informational
    
    def _unwrap_update_item(value: Any) -> Any:
        if isinstance(value, UpdateItem):
            return value.value
        # Be robust to UpdateItem-like wrappers imported through alternate paths.
        if hasattr(value, "value") and hasattr(value, "mode"):
            return getattr(value, "value")
        return value

    # Validate tensor properties
    for spec_name, spec in specifications.items():
        if spec.name in output:
            value = _unwrap_update_item(output[spec.name])
            
            # Check if it's a tensor
            if not isinstance(value, torch.Tensor):
                errors.append(f"{spec.name} must be a torch.Tensor, got {type(value)}")
                continue
            
            # Check for NaN or Inf
            if torch.any(torch.isnan(value)):
                errors.append(f"{spec.name} contains NaN values")
            if torch.any(torch.isinf(value)):
                errors.append(f"{spec.name} contains Inf values")
            
            # Check sign for certain flux types
            if spec.direction == FluxDirection.OUTPUT and spec.flux_type == FluxType.RESPIRATION:
                if torch.any(value < 0):
                    errors.append(f"{spec.name} (respiration) should be non-negative")
    
    return errors


def standardize_output_keys(output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Standardize output dictionary keys to match StateManager expectations.
    
    Parameters:
    -----------
    output : Dict[str, torch.Tensor]
        Raw functor output
        
    Returns:
    --------
    Dict[str, torch.Tensor]
        Standardized output
    """
    # Enforce canonical output keys only; do not remap historical aliases.
    return dict(output)


def create_output_summary(output: Dict[str, torch.Tensor],
                         dt: float) -> Dict[str, float]:
    """
    Create a summary of functor outputs for logging/diagnostics.
    
    Parameters:
    -----------
    output : Dict[str, torch.Tensor]
        Functor output
    dt : float
        Time step size [s]
        
    Returns:
    --------
    Dict[str, float]
        Summary statistics
    """
    summary = {}

    def _scalarize(value: torch.Tensor) -> float:
        if not isinstance(value, torch.Tensor):
            return float(value)
        if value.numel() == 1:
            return _scalar_from_value(value)
        return _scalar_from_value(value.mean())
    
    # Calculate total fluxes over time step
    for key, value in output.items():
        scalar_value = _scalarize(value)
        if 'flux' in key or 'respiration' in key or 'emission' in key:
            # Convert rate to total for time step
            summary[f"{key}_total"] = scalar_value * dt
            summary[f"{key}_rate"] = scalar_value
        elif 'delta' in key:
            # Pool changes
            summary[f"{key}_change"] = scalar_value * dt
        else:
            # Other outputs (states, diagnostics)
            summary[key] = scalar_value
    
    # Calculate net carbon flux
    carbon_outputs = sum(v for k, v in summary.items() 
                        if ('respiration' in k or 'carbon_leaching' in k) and '_total' in k)
    summary['net_carbon_loss'] = carbon_outputs
    
    # Calculate net nitrogen flux
    nitrogen_outputs = sum(v for k, v in summary.items()
                         if ('n2o' in k or 'nitrogen_leaching' in k or 'denitrification' in k) and '_total' in k)
    summary['net_nitrogen_loss'] = nitrogen_outputs
    
    return summary


# Export main components
__all__ = [
    'FluxDirection',
    'FluxType',
    'PoolType',
    'FluxSpecification',
    'PoolSpecification',
    'PedosphereFunctorContract',
    'PedosphereContractMixin',
    'StandardPedosphereOutputs',
    'StandardPedospherePools',
    'get_pedosphere_carbon_pool_names',
    'get_pedosphere_carbon_pool_categories',
    'validate_functor_output',
    'standardize_output_keys',
    'create_output_summary'
]


