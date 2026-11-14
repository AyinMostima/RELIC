"""
Functor Bridge Classes for RELIC Framework

This module implements the bridge classes that unify the FunctorBase execution
shell with the four abstract base classes (PhysicalField, BiologicalProcess,
TransportMechanism, ScaleCoupling). These bridge classes provide the standard
pattern for all executable scientific modules.

Bridge classes solve the inheritance pattern inconsistency by providing:
1. Single inheritance pattern for all executable functors
2. Unified parameter access through FunctorBase
3. Enhanced capability access through abstract base classes
4. Proper MRO (Method Resolution Order) handling
5. Automatic metadata and category assignment
6. Unit consistency validation and conversion

Design Pattern: Bridge Pattern + Multiple Inheritance
Mathematical Foundation: Category Theory Functors with Proper Composition

References:
    - Gang of Four Design Patterns - Bridge Pattern
    - Mac Lane, S. (1998). Categories for the Working Mathematician
    - Python Multiple Inheritance and MRO Documentation
    - CF-1.8 Climate and Forecast Metadata Conventions
"""

import torch
from typing import Any, Optional, Tuple

from torch import Tensor
import logging

from .functor_base import FunctorBase
from .functor_types import ConfigDict, DependencyDeclaration, FunctorMetadata, ProcessCategory
from .functor_bridge_utils import (
    BridgeParameterMixin,
)
from ..abstractions import (
    BiologicalProcess,
    StateDict,
    FluxDict,
)
logger= logging.getLogger(__name__)

class BiologicalProcessFunctor(BridgeParameterMixin, BiologicalProcess, FunctorBase):  # type: ignore[misc,unused-ignore]
    """
    Bridge class providing biological process specific functionality.
    
    This class bridges between the abstract BiologicalProcess interface and
    the concrete FunctorBase implementation, providing biological process
    specific methods and utilities while inheriting the full functor 
    execution framework.
    
    Provides:
        - Temperature response functions (Q10, Arrhenius)  
        - Moisture response functions
        - Nutrient limitation functions
        - Enzyme kinetics utilities
        - Photosynthesis helpers
        - Respiration utilities
        
    Examples:
        >>> class PhotosynthesisFunctor(BiologicalProcessFunctor):
        ...     def forward(self, state, time, dt, **kwargs):
        ...         # Scientific inputs come from the state; parameters from FunctorBase
        ...         vcmax = self.get_parameter_value('vcmax25', required=True)
        ...         return {'gpp': vcmax * state['light_limitation']}
    """
    def get_dependencies(self) -> DependencyDeclaration:
        """
        Return the instance-declared dependencies (bridge default).

        Returns:
            DependencyDeclaration: Names declared through config, set_dependencies
            and add_dependency. A subclass with a static declaration returns
            ``self._merge_declared_dependencies(static)`` instead.
        """
        return self._merge_declared_dependencies()

    
    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[ConfigDict] = None,
        metadata: Optional[FunctorMetadata] = None,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize biological process functor.

        Parameters:
            name : str
                Functor name
            config : Optional[ConfigDict]
                Configuration dictionary
            metadata : Optional[FunctorMetadata]
                Process metadata (auto-generated if None)
            **kwargs : Any
                Additional keyword arguments
        """
        # Handle name
        if name is None:
            name = kwargs.pop('name', 'biological_process')

        # Handle config
        if config is None:
            config = kwargs.pop('config', {})

        # Ensure metadata category is set correctly
        if metadata is None:
            metadata = FunctorMetadata(
                name=name,
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                description=f"Biological process functor: {name}"
            )
        elif metadata.category is None:
            metadata.category = ProcessCategory.BIOLOGICAL_PROCESS

        resolved_device = (
            torch.device(device)
            if device is not None
            else torch.device("cpu")
        )

        # Extract BiologicalProcess-specific parameters from kwargs
        process_name = kwargs.pop('process_name', name)
        allometric_scaling = kwargs.pop('allometric_scaling', None)

        # Initialize BiologicalProcess first (scientific rules layer)
        BiologicalProcess.__init__(
            self,
            process_name=process_name,
            allometric_scaling=allometric_scaling,
            device=resolved_device
        )

        # Initialize FunctorBase (system interaction layer)
        FunctorBase.__init__(
            self,
            name=name,
            config=config,
            metadata=metadata,
            device=resolved_device,
            **kwargs,
        )

        # Initialize additional biological process parameters
        self._init_biological_parameters()
        self._sync_biological_metadata()
        self._sync_conservation_targets_from_metadata()
    
    def _init_biological_parameters(self) -> None:
        """Initialize biological process specific parameters."""
        # Set default biological process parameters if not specified
        if not hasattr(self, 'process_name'):
            self.process_name = self.name
        manager = self.get_parameter_manager()
        default_namespace = self._default_parameter_namespace()
        ph_optimum = getattr(self, 'ph_optimum', None)
        if ph_optimum is None:
            ph_optimum = None
            if (
                manager is not None
                and manager.get_parameter(f"{default_namespace}.ph_optimum") is not None
            ):
                ph_optimum = FunctorBase.get_parameter_value(
                    self,
                    'ph_optimum',
                    required=False,
                )
            if ph_optimum is None:
                ph_optimum = FunctorBase.get_parameter_value(
                    self,
                    'ph_optimum',
                    namespace='validation.numerical.response',
                    required=True,
                )
            self.ph_optimum = ph_optimum

    def _sync_biological_metadata(self) -> None:
        """Update FunctorMetadata with biological defaults."""
        metadata = getattr(self, 'metadata', None)
        if metadata is None:
            return

        if metadata.references is None:
            metadata.references = []

        if not isinstance(metadata.parameters, dict):
            metadata.parameters = {}

        parameter_defaults = {
            'ph_optimum': {
                'description': 'Optimal pH for response curve',
                'units': 'dimensionless',
                'namespace': 'validation.numerical.response',
                'required': False
            },
            'ph_width': {
                'description': 'Width of pH response curve',
                'units': 'dimensionless',
                'namespace': 'validation.numerical.response',
                'required': False
            },
            'moisture_half_saturation': {
                'description': 'Half-saturation point for moisture response',
                'units': 'dimensionless',
                'namespace': 'validation.numerical.response',
                'required': False
            },
            'moisture_linear_min': {
                'description': 'Minimum moisture for linear response',
                'units': 'dimensionless',
                'namespace': 'validation.numerical.response',
                'required': False
            },
            'moisture_linear_max': {
                'description': 'Maximum moisture for linear response',
                'units': 'dimensionless',
                'namespace': 'validation.numerical.response',
                'required': False
            },
            'moisture_optimum': {
                'description': 'Optimum moisture for bell-shaped response',
                'units': 'dimensionless',
                'namespace': 'validation.numerical.response',
                'required': False
            },
            'moisture_width': {
                'description': 'Width for bell-shaped moisture response',
                'units': 'dimensionless',
                'namespace': 'validation.numerical.response',
                'required': False
            }
        }

        for key, spec in parameter_defaults.items():
            if key not in metadata.parameters:
                metadata.parameters[key] = spec.copy()
            else:
                existing = metadata.parameters[key]
                if isinstance(existing, str):
                    metadata.parameters[key] = {'description': existing}
                    existing = metadata.parameters[key]
                if isinstance(existing, dict):
                    existing.setdefault('description', spec['description'])
                    existing.setdefault('units', spec['units'])
                    existing.setdefault('namespace', spec['namespace'])
                    existing.setdefault('required', spec['required'])

        if metadata.conservation_laws:
            if isinstance(metadata.conservation_laws, list):
                metadata.conservation_laws = {
                    law: {'quantity': law} for law in metadata.conservation_laws
                }
        else:
            metadata.conservation_laws = {}

    def get_parameter_value(  # type: ignore[override,unused-ignore]
        self,
        param_name: Optional[str] = None,
        *,
        name: Optional[str] = None,
        units: Optional[str] = None,
        bounds: Optional[Tuple[float, float]] = None,
        validate: bool = True,
        namespace: Optional[str] = None,
        source_order: Tuple[str, ...] = ("manager", "config"),
        expected_type: Optional[type] = None,
        required: bool = False,
        **kwargs: Any,
    ) -> Any:
        """MRO disambiguation: resolve to FunctorBase, not BiologicalProcess.

        Without this override the C3 MRO resolves to
        BiologicalProcess.get_parameter_value (2-source: config+manager)
        instead of FunctorBase.get_parameter_value (full multi-source
        resolution with namespace, caching, and validation support).
        Do NOT remove without verifying MRO resolution order.
        """
        resolved_name = param_name if param_name is not None else name
        if resolved_name is None:
            raise TypeError("get_parameter_value requires 'param_name' or keyword 'name'.")

        return FunctorBase.get_parameter_value(
            self,
            name=resolved_name,
            units=units,
            bounds=bounds,
            validate=validate,
            namespace=namespace,
            source_order=source_order,
            expected_type=expected_type,
            required=required,
            **kwargs,
        )

    def get_metadata(self) -> Optional[FunctorMetadata]:
        """Get metadata for this functor."""
        return getattr(self, 'metadata', None)


    def validate_biological_units(self, fluxes: FluxDict) -> None:
        """
        Validate that biological process outputs have consistent units.
        
        Parameters:
            fluxes : FluxDict
                Output fluxes to validate
        """
        # Basic unit validation - can be extended
        for key, value in fluxes.items():
            if not isinstance(value, Tensor):
                logger.warning(f"Non-tensor output in biological process {self.name}: {key}")

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """
        Primary computation entry point for biological functors.

        Concrete biological functors must implement this method directly.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward(state, time, dt, **kwargs)."
        )

__all__ = ['BiologicalProcessFunctor']
