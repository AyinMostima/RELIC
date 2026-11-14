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
from typing import Dict, Any, Optional, Union

from torch import Tensor
import logging

from .functor_base import FunctorBase
from .functor_types import ConfigDict, DependencyDeclaration, FunctorMetadata, ProcessCategory
from .functor_bridge_utils import (
    BridgeParameterMixin,
)
from ..abstractions import (
    PhysicalField,
    FluxDict,
)

logger = logging.getLogger(__name__)

class PhysicalFieldFunctor(BridgeParameterMixin, PhysicalField, FunctorBase):  # type: ignore[misc,unused-ignore]
    """
    Bridge Class: Physical Field Functor
    
    Executable functor for physical field evolution including temperature
    fields, concentration distributions, electromagnetic fields, and other
    spatially distributed physical quantities.
    
    Combines FunctorBase execution shell with PhysicalField capabilities:
    - Spatial interpolation and gradient calculation
    - Field evolution and update mechanisms
    - Grid-based and mesh-free computations
    - Boundary condition management
    
    A concrete class declares SPATIAL_DIMS (1, 2, 3 or None for a point
    model) and implements forward and get_value_at (a point model raises
    NotImplementedError there); the field's identity and units are its
    declared outputs and their contracts.

    Usage Example:
        >>> class TemperatureField(PhysicalFieldFunctor):
        ...     SPATIAL_DIMS = 2
        ...     def forward(self, state, time, dt, **kwargs):
        ...         gradient = self.compute_gradient(coordinates)
        ...         return {'temperature_update': field_change}
        ...     def get_value_at(self, coordinates):
        ...         ...
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
        **kwargs: Any,
    ) -> None:
        """
        Initialize physical field functor.

        Parameters:
            name : str
                Functor name
            config : Optional[ConfigDict]
                Configuration dictionary
            metadata : Optional[FunctorMetadata]
                Process metadata
            **kwargs : Any
                Additional keyword arguments
        """
        # Handle name
        if name is None:
            name = kwargs.pop('name', 'physical_field')

        # Handle config
        if config is None:
            config = kwargs.pop('config', {})

        # Ensure metadata category
        if metadata is None:
            metadata = FunctorMetadata(
                name=name,
                category=ProcessCategory.PHYSICAL_FIELD,
                description=f"Physical field functor: {name}"
            )
        elif metadata.category is None:
            metadata.category = ProcessCategory.PHYSICAL_FIELD

        # The field declares SPATIAL_DIMS as a class attribute; its identity and
        # units are its declared outputs and their contracts.
        for keyword in ("field_name", "units", "default_value", "spatial_dims", "temporal_dependence"):
            if keyword in kwargs:
                raise TypeError(
                    f"{type(self).__name__}: '{keyword}' is not a constructor keyword; a "
                    "PhysicalFieldFunctor declares SPATIAL_DIMS as a class attribute and takes "
                    "its identity and units from its declared outputs and their contracts"
                )
        device = kwargs.pop('device', None)
        resolved_device = (
            torch.device(device)
            if device is not None
            else (torch.device("cpu"))
        )

        # Initialize PhysicalField first (scientific rules layer)
        PhysicalField.__init__(self, device=resolved_device)

        # Initialize FunctorBase (system interaction layer)
        FunctorBase.__init__(
            self,
            name=name,
            config=config,
            metadata=metadata,
            device=resolved_device,
            **kwargs,
        )

        self._sync_field_metadata()
        self._sync_conservation_targets_from_metadata()
    
    def _sync_field_metadata(self) -> None:
        """Update FunctorMetadata with physical field defaults."""
        metadata = getattr(self, 'metadata', None)
        if metadata is None:
            return

        if metadata.references is None:
            metadata.references = []
        field_refs = [
            "Arakawa, A. and Lamb, V. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model.",
            "Randall, D.A. (1994). Geostrophic adjustment and the problem of available potential energy."
        ]
        for ref in field_refs:
            if ref not in metadata.references:
                metadata.references.append(ref)

        if metadata.conservation_laws:
            if isinstance(metadata.conservation_laws, list):
                metadata.conservation_laws = {
                    law: {'quantity': law} for law in metadata.conservation_laws
                }
        else:
            metadata.conservation_laws = {}

    def compute_gradient(self, coordinates: Any) -> Tensor:
        """
        Default implementation of abstract method from PhysicalField.

        Compute field gradient at specified coordinates.

        References:
            - Standard vector calculus: gradient of a constant field is zero.

        Mathematical Expression:
            - For a spatially constant scalar field f(x) = c, grad(f) = 0.

        Parameters:
            coordinates : Any
                Spatial coordinates

        Returns:
            Tensor: Gradient tensor
        """
        if isinstance(coordinates, Tensor):
            coord_tensor = coordinates
        else:
            device = getattr(self, "device", torch.device("cpu"))
            coord_tensor = torch.as_tensor(coordinates, device=device, dtype=torch.float32)

        return torch.zeros_like(coord_tensor)

    def interpolate(
                    self,
                    source_coordinates: Any,
                    target_coordinates: Any) -> Tensor:
        """
        Default implementation of abstract method from PhysicalField.

        Perform interpolation between coordinate systems.

        Parameters:
            source_coordinates : Any
                Source grid coordinates
            target_coordinates : Any
                Target grid coordinates

        Returns:
            Tensor: Interpolated values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement interpolate() with an "
            f"appropriate spatial interpolation scheme"
        )

    def update_field(
                     self,
                     time_step: float,
                     source_terms: Dict[str, Any],
                     boundary_conditions: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Default implementation of abstract method from PhysicalField.

        Advance the field in time via the FunctorBase execution pathway.

        Parameters:
            time_step : float
                Time step size in seconds
            source_terms : Dict[str, Any]
                Source or sink terms
            boundary_conditions : Optional[Dict[str, Tensor]]
                Boundary condition specifications
        """
        # References: FunctorBase forward adapter for PhysicalField updates.
        # Mathematical expression: field_update = forward(state, time, dt).
        # Parameter meanings: time_step [s], source_terms [dict], boundary_conditions [dict].
        if time_step <= 0:
            raise ValueError("time_step must be positive")
        if isinstance(source_terms, dict):
            state = dict(source_terms)
        else:
            state = {"source_terms": source_terms}
        kwargs: Dict[str, Any] = {}
        if boundary_conditions is not None:
            kwargs["boundary_conditions"] = boundary_conditions
        updates = self(state, time=0.0, dt=time_step, **kwargs)
        self._last_field_update = updates
        return None

    def compute_laplacian(self, coordinates: Any) -> Tensor:
        """
        Default implementation of abstract method from PhysicalField.

        Compute Laplacian at specified coordinates.

        Parameters:
            coordinates : Any
                Spatial coordinates

        Returns:
            Tensor: Laplacian values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement compute_laplacian() or "
            f"provide Laplacian values via forward()"
        )
    
    def update_field_to_flux(
        self,
        field_update: Tensor,
        dt: Union[float, Tensor],
        field_name: str,
    ) -> FluxDict:
        """
        Convert a field increment over dt to the flux of a declared output.

        References:
            - LeVeque, R. J. (2007). Finite Difference Methods for Ordinary and
              Partial Differential Equations (increment and rate of one step).
        Mathematical Expression:
            F = Delta / dt
        Parameter meanings:
            Delta the field increment over the step, dt the step length, F the rate.
        Units:
            Delta in the field units, dt [s], F in the field units per second.
        Domain/constraints:
            dt > 0; field_name names the declared output that carries F.
        Sign convention:
            F has the sign of Delta (positive increases the field).

        Parameters:
            field_update : Tensor
                Field increment over the step
            dt : float | Tensor
                Time step in seconds
            field_name : str
                Declared output key of the flux

        Returns:
            FluxDict: {field_name: field_update / dt}
        """
        dt_tensor = dt if isinstance(dt, Tensor) else torch.as_tensor(float(dt), device=field_update.device)
        return {field_name: field_update / dt_tensor}

__all__ = ['PhysicalFieldFunctor']
