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
from typing import Dict, Any, Optional, Union, Tuple

from torch import Tensor
import logging

from .functor_base import FunctorBase
from .functor_types import ConfigDict, DependencyDeclaration, FunctorMetadata, ProcessCategory
from .functor_bridge_utils import (
    BridgeParameterMixin,
    _ensure_tensor,
)
from ..abstractions import (
    ScaleCoupling,
    StateDict,
    FluxDict,
)
from ...parameters.namespaces import resolve_namespace

logger = logging.getLogger(__name__)

class ScaleCouplingFunctor(BridgeParameterMixin, ScaleCoupling, FunctorBase):  # type: ignore[misc,unused-ignore]
    """
    Bridge Class: Scale Coupling Functor
    
    Executable functor for scale transformation processes including
    upscaling, downscaling, and cross-scale information transfer.
    
    Combines FunctorBase execution shell with ScaleCoupling capabilities:
    - Conservative upscaling and downscaling
    - Resolution conversion between grids
    - Hierarchical scale transformations
    - Conservation law enforcement across scales
    
    Usage Example:
        >>> class SpatialUpscaling(ScaleCouplingFunctor):
        ...     def forward(self, state, time, dt, **kwargs):
        ...         # Use ScaleCoupling methods
        ...         coarse_data = self.upscale(fine_data)
        ...         return {'upscaled_field': coarse_data}
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
        Initialize scale coupling functor.

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
            name = kwargs.pop('name', 'coupling.scales')
        normalized_name = name.replace(".", "_")
        if normalized_name != name:
            name = normalized_name

        # Handle config
        if config is None:
            config = kwargs.pop('config', {})

        # Ensure metadata category
        if metadata is None:
            metadata = FunctorMetadata(
                name=name,
                category=ProcessCategory.SCALE_COUPLING,
                description=f"Scale coupling functor: {name}",
                references=[
                    "Jones, P.W. (1999). First- and second-order conservative remapping schemes.",
                    "Shepard, D. (1968). Two-dimensional interpolation function for irregularly spaced data.",
                ],
            )
        else:
            if metadata.category is None:
                metadata.category = ProcessCategory.SCALE_COUPLING
            if getattr(metadata, "references", None) in (None, []):
                metadata.references = [
                    "Jones, P.W. (1999). First- and second-order conservative remapping schemes.",
                    "Shepard, D. (1968). Two-dimensional interpolation function for irregularly spaced data.",
                ]
            if getattr(metadata, "name", None) is None:
                metadata.name = name
            elif "." in metadata.name:
                metadata.name = metadata.name.replace(".", "_")

        # Extract ScaleCoupling-specific parameters. The scale pair and the
        # coupling type are declared by the concrete coupling (they select the
        # scale operation and the parameter namespace); none is fabricated.
        missing_scales = [
            key for key in ("source_scale", "target_scale", "coupling_type") if key not in kwargs
        ]
        if missing_scales:
            raise TypeError(
                f"{type(self).__name__} passes no {missing_scales} to ScaleCouplingFunctor; "
                "declare source_scale, target_scale and coupling_type"
            )
        source_scale = kwargs.pop('source_scale')
        target_scale = kwargs.pop('target_scale')
        coupling_type = kwargs.pop('coupling_type')
        conservation_properties = kwargs.pop('conservation_properties', None)
        device = kwargs.pop('device', None)
        resolved_device = (
            torch.device(device)
            if device is not None
            else (torch.device("cpu"))
        )

        # Initialize FunctorBase first (system interaction layer)
        # This ensures _dependencies and other FunctorBase attributes are properly set
        FunctorBase.__init__(
            self,
            name=name,
            config=config,
            metadata=metadata,
            device=resolved_device,
            **kwargs,
        )

        # Initialize ScaleCoupling second (scientific rules layer)
        ScaleCoupling.__init__(
            self,
            source_scale=source_scale,
            target_scale=target_scale,
            coupling_type=coupling_type,
            conservation_properties=conservation_properties,
            device=resolved_device
        )

        # Initialize additional coupling parameters
        self._init_coupling_parameters()
        self._sync_scale_metadata()
        self._sync_conservation_targets_from_metadata()

    @staticmethod
    def _to_scalar(value: Union[torch.Tensor, float, int]) -> float:
        """
        Convert tensor or numeric value to Python float.
        """
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _ensure_tensor(
        self,
        value: Any,
        dtype: Optional[torch.dtype] = None,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert inputs to tensors while aligning dtype/device with an optional reference tensor.

        Notes
        -----
        This helper centralizes tensor casting for coupling functors to avoid duplicated
        implementations in downstream modules.
        """
        target_device = (
            reference.device if reference is not None else getattr(self, "device", torch.device("cpu"))
        )
        target_dtype = reference.dtype if reference is not None else (dtype or torch.float32)
        if isinstance(value, torch.Tensor):
            return value.to(device=target_device, dtype=target_dtype)
        return torch.tensor(float(value), device=target_device, dtype=target_dtype)

    def get_namespaced_parameter(
        self,
        key: str,
        namespace: Any,
        *,
        units: Optional[str] = None,
        as_tensor: bool = False,
        as_scalar: bool = False,
        required: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Retrieve a parameter from a specific namespace with consistent validation.

        Parameters
        ----------
        key : str
            Parameter key without namespace prefix.
        namespace : Any
            ParameterNamespace enum or namespace string.
        units : Optional[str]
            Expected units for validation.
        as_tensor : bool
            Return value as tensor aligned to functor device.
        as_scalar : bool
            Return value as Python float.
        required : bool
            Whether to raise if the parameter is missing.
        """
        resolved_ns = resolve_namespace(namespace)

        # required=False covers absence (None); lookup failures propagate.
        value = self.get_parameter_value(
            key,
            namespace=resolved_ns,
            units=units,
            validate=True,
            required=required,
            **kwargs,
        )

        if as_tensor:
            return self._ensure_tensor(value)
        if as_scalar:
            return self._to_scalar(value)
        return value

    def _init_coupling_parameters(self) -> None:
        """Check the scale-coupling attributes set by ScaleCoupling.__init__.

        ``source_scale``, ``target_scale`` and ``coupling_type`` come only from
        ``ScaleCoupling.__init__``; the bridge fabricates none of them.
        """
        missing = [
            name for name in ("source_scale", "target_scale", "coupling_type")
            if not hasattr(self, name)
        ]
        if missing:
            raise RuntimeError(
                f"{type(self).__name__} has no {missing}: ScaleCoupling.__init__ was "
                "bypassed, and the bridge does not fabricate scale-coupling attributes"
            )

    def _sync_scale_metadata(self) -> None:
        """Update FunctorMetadata with scale coupling defaults."""
        metadata = getattr(self, 'metadata', None)
        if metadata is None:
            return

        if metadata.references is None:
            metadata.references = []
        scale_refs = [
            "Pielke, R.A. (2002). Mesoscale meteorological modeling.",
            "Leung, L.R. and Ghan, S.J. (1999). Pacific Northwest climate sensitivity to global warming."
        ]
        for ref in scale_refs:
            if ref not in metadata.references:
                metadata.references.append(ref)

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
        """MRO disambiguation: resolve to FunctorBase, not ScaleCoupling.

        Without this override the C3 MRO resolves to
        ScaleCoupling.get_parameter_value (2-source: config+manager)
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

    def upscale(self, fine_data: Tensor) -> Tensor:
        """
        Resolve an explicit upscale payload from ``forward(...)``.

        A missing payload is a contract error because silently returning the
        input would bypass the declared scale transformation.

        Parameters:
            fine_data : Tensor
                Fine-scale data to be upscaled

        Returns:
            Tensor: Coarse-scale data
        """
        state = {'fine_data': fine_data, 'operation': 'upscale'}
        result = self.forward(state, 0.0, 1.0)

        if isinstance(result, dict):
            for key, value in result.items():
                lowered = key.lower()
                if 'upscale' in lowered or 'coarse' in lowered:
                    if isinstance(value, Tensor):
                        return value
                    return _ensure_tensor(value, getattr(self, 'device', None))
        raise RuntimeError(
            f"Scale coupling '{self.name}' did not return an explicit upscale payload."
        )

    def downscale(self, coarse_data: Tensor) -> Tensor:
        """
        Resolve an explicit downscale payload from ``forward(...)``.

        A missing payload is a contract error because silently returning the
        input would bypass the declared scale transformation.

        Parameters:
            coarse_data : Tensor
                Coarse-scale data to be downscaled

        Returns:
            Tensor: Fine-scale data
        """
        state = {'coarse_data': coarse_data, 'operation': 'downscale'}
        result = self.forward(state, 0.0, 1.0)

        if isinstance(result, dict):
            for key, value in result.items():
                lowered = key.lower()
                if 'downscale' in lowered or 'fine' in lowered:
                    if isinstance(value, Tensor):
                        return value
                    return _ensure_tensor(value, getattr(self, 'device', None))
        raise RuntimeError(
            f"Scale coupling '{self.name}' did not return an explicit downscale payload."
        )

    def check_conservation_bridge(self, input_data: Any, output_data: Any) -> Any:
        """
        Check conservation laws across scale transformation.

        Mathematical Expression:
            - Residual (per conserved quantity): r = output - input
            - Aggregate check (tensor): |sum(output) - sum(input)| / (|sum(input)| + eps) < tol
        References:
            - Byun (1999), multiscale atmospheric mass-consistency diagnostics.
            - RELIC coupling conservation policy in coupling.scales.defaults.
        Parameter meanings:
            - epsilon: dimensionless denominator safeguard.
            - conservation_tolerance: dimensionless acceptance threshold.
        Units and route:
            - Relative error, epsilon, and tolerance are dimensionless.
            - This check is diagnostic and does not update conserved state.
        
        Parameters:
            input_data : Any
                Input data before transformation (Tensor or dict[str, Tensor])
            output_data : Any
                Output data after transformation (Tensor or dict[str, Tensor])
                
        Returns:
            Any:
                - If input/output are dicts: dict[str, float] residuals (output - input) per key.
                - Otherwise: bool, True if aggregate conservation is satisfied.
        """
        try:
            if isinstance(input_data, dict) and isinstance(output_data, dict):
                violations: Dict[str, float] = {}
                keys = set(input_data.keys()) | set(output_data.keys())
                for key in keys:
                    before = input_data.get(key, torch.tensor(0.0))
                    after = output_data.get(key, torch.tensor(0.0))
                    before_tensor = before if isinstance(before, Tensor) else torch.as_tensor(float(before))
                    after_tensor = after if isinstance(after, Tensor) else torch.as_tensor(float(after))
                    violations[key] = float((after_tensor - before_tensor).detach().cpu().item())
                return violations

            input_tensor = input_data if isinstance(input_data, Tensor) else torch.as_tensor(input_data)
            output_tensor = output_data if isinstance(output_data, Tensor) else torch.as_tensor(output_data)

            input_sum = torch.sum(input_tensor)
            output_sum = torch.sum(output_tensor)

            epsilon = FunctorBase.get_parameter_value(
                self,
                name="epsilon",
                namespace="coupling.scales.defaults",
                units="dimensionless",
                source_order=("manager",),
                validate=True,
                required=True,
            )
            relative_error = torch.abs(input_sum - output_sum) / (
                torch.abs(input_sum) + float(epsilon)
            )

            conservation_tolerance = FunctorBase.get_parameter_value(
                self,
                name="conservation_tolerance",
                namespace="coupling.scales.defaults",
                units="dimensionless",
                source_order=("manager",),
                validate=True,
                required=True,
            )
            is_conserved = relative_error < float(conservation_tolerance)

            if not is_conserved:
                logger.warning(
                    f"Conservation violation in {self.name}: relative error = {relative_error:.2e}"
                )

            return bool(is_conserved)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            logger.error(f"Conservation check failed in {self.name}: {e}")
            raise

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """
        Forward method implementation for scale coupling operations.

        This method provides a default implementation that can perform
        upscaling or downscaling based on the configuration.

        Parameters:
            state : StateDict
                Current system state dictionary
            time : float
                Current simulation time
            dt : float
                Time step size
            **kwargs : Any
                Additional keyword arguments

        Returns:
            FluxDict
                Dictionary of scale transformation results
        """
        # Default implementation - return empty updates
        # Specific scale coupling functors should override this method
        return {"scale_update": torch.zeros(1, device=self.device)}

__all__ = ['ScaleCouplingFunctor']
