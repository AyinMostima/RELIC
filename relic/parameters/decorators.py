"""
Parameter Management Decorators for RELIC

This module provides decorators for automatic parameter registration,
validation, and management in Functor classes.

These decorators simplify parameter handling by automatically registering
parameters with the global manager and ensuring consistent validation.

Key Features:
    - Automatic parameter registration on initialization
    - Parameter validation enforcement
    - Namespace management
    - Gradient tracking for learnable parameters

Design Patterns:
    - Decorator Pattern for behavior enhancement
    - Aspect-Oriented Programming for cross-cutting concerns

References:
    - Gamma, E. et al. (1994). Design Patterns - Decorator Pattern
    - Python PEP 318 - Decorators for Functions and Methods
"""

import functools
from typing import Dict, Any, Optional, Union, List, Callable, Type
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from torch import Tensor

# Import parameter management components
from .manager import ParameterSpec, ParameterCategory, get_global_manager

logger = logging.getLogger(__name__)

# Type aliases
ParameterDict = Dict[str, Union[float, int, str, bool, Tensor]]


@dataclass
class ParameterDefinition:
    """
    Parameter definition for decorator configuration.
    
    Attributes:
        name : str
            Parameter name
        default : Any
            Default value
        units : str
            Physical units
        bounds : tuple
            (min, max) bounds
        description : str
            Parameter description
        requires_grad : bool
            Whether parameter needs gradients
    """
    name: str
    default: Any
    units: str  # required: no universal unit
    bounds: tuple  # required: (-inf, inf) declares an unbounded parameter
    description: str = ""
    requires_grad: bool = False
    validate: bool = False
    register: bool = False


def managed_parameters(namespace: Optional[str] = None,
                      register_missing: bool = True,
                      validate_on_init: bool = True,
                      sync_with_manager: bool = True):
    """
    Class decorator for automatic parameter management.
    
    Registers the declared class parameters with the global parameter
    manager and handles synchronization between local and managed parameters.
    A managed parameter is an attribute ``<name>`` together with a
    ``<name>_spec`` mapping that states ``units``, ``bounds``, ``dtype`` and
    ``description``; an incomplete spec raises ValueError, and an attribute
    without a spec is not a managed parameter.

    Parameters:
        namespace : Optional[str]
            Parameter namespace (auto-generated if None)
        register_missing : bool
            Register parameters not in manager
        validate_on_init : bool
            Validate parameters on initialization
        sync_with_manager : bool
            Keep parameters synchronized with manager

    Returns:
        Callable: Decorated class

    Examples:
        >>> @managed_parameters(namespace="biosphere.photosynthesis")
        ... class PhotosynthesisFunctor(FunctorBase):
        ...     def __init__(self):
        ...         self.vcmax25 = 60.0
        ...         self.vcmax25_spec = {"units": "umol m-2 s-1", "bounds": (0.0, 200.0),
        ...                              "dtype": "float32", "description": "Vcmax at 25 degC"}
    """
    def decorator(cls: Type) -> Type:
        """Inner decorator function."""
        original_init = cls.__init__
        
        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            """Wrapped initialization with parameter management."""
            # Call original init
            original_init(self, *args, **kwargs)
            
            # Determine namespace
            param_namespace = namespace
            if param_namespace is None:
                # Auto-generate from class hierarchy
                module = cls.__module__
                class_name = cls.__name__
                if hasattr(self, 'metadata') and hasattr(self.metadata, 'category'):
                    category = self.metadata.category.value
                    param_namespace = f"{category}.{class_name.lower()}"
                else:
                    param_namespace = f"{module.split('.')[-1]}.{class_name.lower()}"
            
            # Get parameter manager
            manager = get_global_manager()
            
            # Ensure manager is initialized
            if hasattr(manager, 'state'):
                from ..core.managers.base import ManagerState
                if manager.state != ManagerState.READY:
                    # Initialize if not ready
                    if manager.state == ManagerState.UNINITIALIZED:
                        manager.initialize()
                    # For INITIALIZING state, wait or continue
            
            # Register parameters
            if register_missing:
                _register_class_parameters(self, manager, param_namespace)
            
            # Sync with manager
            if sync_with_manager:
                _sync_parameters_from_manager(self, manager, param_namespace)
            
            # Validate if requested
            if validate_on_init:
                _validate_class_parameters(self, manager, param_namespace)
            
            # Set up synchronization hooks if using nn.Parameter
            if sync_with_manager:
                _setup_parameter_hooks(self, manager, param_namespace)
            
            logger.debug(f"Initialized managed parameters for {cls.__name__}")
        
        cls.__init__ = wrapped_init
        cls._parameter_namespace = namespace
        cls._managed_parameters = True
        
        return cls
    
    return decorator


def parameter(name: str,
             default: Any = None,
             *,
             units: str,
             bounds: tuple,
             description: str = "",
             requires_grad: bool = False,
             validate: bool = False,
             register: bool = False,
             dtype: Optional[str] = None):
    """
    Property decorator for managed parameters.
    
    Creates a property that automatically syncs with the parameter manager
    and validates on set operations.
    
    Parameters:
        name : str
            Parameter name
        default : Any
            Default value
        units : str
            Physical units (required, keyword-only)
        bounds : tuple
            (min, max) bounds (required, keyword-only; (-inf, inf) declares
            an unbounded parameter)
        description : str
            Parameter description
        requires_grad : bool
            Whether parameter needs gradients
        validate : bool
            Whether to validate parameter bounds on set
        register : bool
            Whether to register parameter with global manager
        dtype : Optional[str]
            Data type of the registered ParameterSpec; required when
            ``register`` is True

    Returns:
        property: Managed parameter property

    Examples:
        >>> class MyFunctor(FunctorBase):
        ...     @parameter(name="rate", default=1.0, units="1/s", bounds=(0, 10))
        ...     def reaction_rate(self):
        ...         return self._reaction_rate
    """
    if register and dtype is None:
        raise ValueError(
            f"parameter({name!r}, register=True) requires a declared dtype for the registered ParameterSpec"
        )

    class ManagedProperty(property):
        """Property subclass that can store metadata."""
        def __init__(self, getter, setter=None, deleter=None, doc=None):
            super().__init__(getter, setter, deleter, doc)
            self.parameter_metadata = None
    
    def decorator(func: Callable) -> ManagedProperty:
        """Inner decorator function."""
        attr_name = f"_{name}"
        
        def getter(self):
            """Get parameter value."""
            if not hasattr(self, attr_name):
                # Initialize from manager or default
                manager = get_global_manager()
                namespace = getattr(self, '_parameter_namespace', '')
                full_key = f"{namespace}.{name}" if namespace else name
                
                # Auto-register if specified
                if register and not manager.has_parameter(full_key):
                    from .manager import ParameterSpec
                    spec = ParameterSpec(
                        key=full_key,
                        value=default,
                        units=units,
                        dtype=dtype,
                        bounds=bounds,
                        category=ParameterCategory.MODEL_PARAMETER,
                        description=description
                    )
                    manager.register_parameter(spec)
                
                value = manager.get(full_key, default=default)
                setattr(self, attr_name, value)
            
            return getattr(self, attr_name)
        
        def setter(self, value):
            """Set parameter value with validation."""
            # Validate bounds if validate is True
            if validate and isinstance(value, (int, float, Tensor)):
                numeric_value = float(value) if not isinstance(value, Tensor) else float(value.item())
                if not (bounds[0] <= numeric_value <= bounds[1]):
                    raise ValueError(
                        f"Parameter {name} value {numeric_value} outside bounds {bounds}"
                    )
            
            # Update local value
            setattr(self, attr_name, value)
            
            # Sync with manager
            manager = get_global_manager()
            namespace = getattr(self, '_parameter_namespace', '')
            full_key = f"{namespace}.{name}" if namespace else name
            
            manager.set(full_key, value, validate=validate, notify=True)
            
            logger.debug(f"Updated parameter {full_key} = {value}")
        
        # Create managed property
        prop = ManagedProperty(getter, setter)
        
        # Store metadata
        prop.parameter_metadata = ParameterDefinition(
            name=name,
            default=default,
            units=units,
            bounds=bounds,
            description=description,
            requires_grad=requires_grad,
            validate=validate,
            register=register
        )
        
        return prop
    
    return decorator


def validated_parameter(name: str,
                       validator: Union[Callable[[Any], bool], List[Callable[[Any], bool]]]):
    """
    Decorator for parameters with custom validation.

    Parameters:
        name : str
            Parameter name
        validator : Union[Callable[[Any], bool], List[Callable[[Any], bool]]]
            Validation function(s) returning True if valid

    Returns:
        Callable: Parameter decorator with validation
        
    Examples:
        >>> @validated_parameter(name="rate", validator=lambda x: 0 < x < 1)
        ... def probability_param(self):
        ...     return self._prob
    """
    class ValidatedProperty(property):
        """Property with validation that can be extended with custom setter."""
        def __init__(self, getter, setter=None, deleter=None, doc=None):
            super().__init__(getter, setter, deleter, doc)
            self._validators = validator if isinstance(validator, list) else [validator]
            self._name = name
        
        def setter(self, setter_func):
            """Override setter to add validation."""
            def validated_setter(instance, value):
                # Run validation first
                for val_func in self._validators:
                    if not val_func(value):
                        raise ValueError(f"Validation failed for {self._name}: {value}")
                # Then call the user's setter
                return setter_func(instance, value)
            
            return ValidatedProperty(self.fget, validated_setter, self.fdel, self.__doc__)
    
    def decorator(func: Callable) -> ValidatedProperty:
        """Inner decorator function."""
        return ValidatedProperty(func)
    
    return decorator


def learnable_parameter(name: str,
                       initial_value: Union[float, Tensor]):
    """
    Decorator for learnable parameters (nn.Parameter).

    Creates a PyTorch parameter that can be optimized through gradients
    while maintaining registration with the parameter manager.

    Parameters:
        name : str
            Parameter name
        initial_value : Union[float, Tensor]
            Initial parameter value

    Returns:
        property: Property that returns the nn.Parameter

    Examples:
        >>> class MyModule(nn.Module):
        ...     @learnable_parameter("weight", 1.0)
        ...     def weight(self):
        ...         return self._weight
    """
    def decorator(func: Callable) -> property:
        """Inner decorator function."""
        attr_name = f"_{name}"
        
        def getter(self):
            """Get learnable parameter."""
            if not hasattr(self, attr_name):
                # Create nn.Parameter on first access
                if isinstance(initial_value, Tensor):
                    tensor_value = initial_value.clone()
                else:
                    tensor_value = torch.tensor(float(initial_value))
                
                param = nn.Parameter(tensor_value)
                setattr(self, attr_name, param)
                
                # Register with module if it's a nn.Module
                if hasattr(self, 'register_parameter'):
                    self.register_parameter(name, param)
            
            return getattr(self, attr_name)
        
        return property(getter)
    
    return decorator


# Helper functions for decorator implementation

def _register_class_parameters(instance: Any,
                              manager: Any,
                              namespace: str) -> None:
    """
    Register all class parameters with manager.
    
    Parameters:
        instance : Any
            Class instance
        manager : Any
            Parameter manager
        namespace : str
            Parameter namespace
    """
    # A managed parameter is an attribute <name> that declares a <name>_spec
    # mapping. The spec states the metadata that has no universal value; nothing
    # is invented for an undeclared attribute or a missing spec field.
    spec_attributes = {}
    for attr_name in dir(instance):
        if attr_name.startswith('_'):
            continue
        attr_value = getattr(instance, attr_name)
        if attr_name.endswith('_spec') and isinstance(attr_value, dict):
            spec_attributes[attr_name[:-5]] = attr_value

    class_name = instance.__class__.__name__
    required_fields = ("units", "bounds", "dtype", "description")
    parameters_to_register = {}
    for param_name, spec_data in spec_attributes.items():
        missing = [name for name in required_fields if name not in spec_data]
        if missing:
            raise ValueError(
                f"{class_name}.{param_name}_spec does not declare {missing}; a managed "
                f"parameter states {list(required_fields)}"
            )
        if not hasattr(instance, param_name):
            raise ValueError(
                f"{class_name}.{param_name}_spec declares a parameter that the instance does not hold"
            )
        parameters_to_register[param_name] = getattr(instance, param_name)

    # Register the declared parameters with their specs
    for attr_name, attr_value in parameters_to_register.items():
        # Check if it's already registered
        full_key = f"{namespace}.{attr_name}"
        if not manager.exists(full_key):
            spec_data = spec_attributes[attr_name]

            # Create parameter spec
            spec = ParameterSpec(
                key=full_key,
                value=attr_value,
                units=spec_data["units"],
                dtype=spec_data["dtype"],
                bounds=spec_data["bounds"],
                description=spec_data["description"],
                category=ParameterCategory.MODEL_PARAMETER
            )
            
            # Handle nn.Parameter specially
            if isinstance(attr_value, nn.Parameter):
                spec.requires_grad = True
                spec.category = ParameterCategory.CALIBRATION_PARAMETER
            
            manager.register_parameter(spec)
            logger.debug(f"Auto-registered parameter {full_key} with spec {spec_data}")


def _sync_parameters_from_manager(instance: Any,
                                 manager: Any,
                                 namespace: str) -> None:
    """
    Sync parameter values from manager to instance.
    
    Parameters:
        instance : Any
            Class instance
        manager : Any
            Parameter manager
        namespace : str
            Parameter namespace
    """
    # Get all parameters in namespace
    param_keys = manager.list(namespace=namespace)
    
    for key in param_keys:
        # Extract attribute name
        attr_name = key.split('.')[-1]
        
        # Get value from manager
        value = manager.get(key)
        
        if value is not None and hasattr(instance, attr_name):
            # Update instance attribute
            current_value = getattr(instance, attr_name)
            
            # Handle nn.Parameter specially
            if isinstance(current_value, nn.Parameter):
                with torch.no_grad():
                    current_value.data.copy_(torch.tensor(float(value)))
            else:
                setattr(instance, attr_name, value)
            
            logger.debug(f"Synced parameter {attr_name} from manager")


def _validate_class_parameters(instance: Any,
                              manager: Any,
                              namespace: str) -> None:
    """
    Validate all class parameters.
    
    Parameters:
        instance : Any
            Class instance
        manager : Any
            Parameter manager
        namespace : str
            Parameter namespace
            
    Raises:
        ValueError: If validation fails
    """
    # Get all parameters in namespace
    param_keys = manager.list(namespace=namespace)
    
    errors = []
    for key in param_keys:
        spec = manager.get_parameter(key)
        if spec:
            value = manager.get(key)
            spec_errors = spec.validate_value(value)
            if spec_errors:
                errors.extend([f"{key}: {error}" for error in spec_errors])
    
    if errors:
        raise ValueError("Parameter validation failed:\n" + "\n".join(errors))


def _setup_parameter_hooks(instance: Any,
                         manager: Any,
                         namespace: str) -> None:
    """
    Set up hooks for parameter synchronization.
    
    Parameters:
        instance : Any
            Class instance
        manager : Any
            Parameter manager
        namespace : str
            Parameter namespace
    """
    # For nn.Module instances, set up forward hooks
    if isinstance(instance, nn.Module):
        def sync_hook(module, input):
            """Pre-forward hook to sync parameters."""
            _sync_parameters_from_manager(module, manager, namespace)
        
        instance.register_forward_pre_hook(sync_hook)
        logger.debug(f"Set up parameter sync hook for {instance.__class__.__name__}")
