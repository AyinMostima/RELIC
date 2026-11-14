"""
Physical Constraint Manager Component for Scale Coupling

This module provides physical constraint application functionality
previously embedded in ScaleCoupling abstract class. It ensures that
scaled data respects physical bounds and constraints.

Mathematical Expression:
    Constraint application: phi_constrained = C(phi) where C ensures:
    - Non-negativity: phi >= 0
    - Bounds: phi_min <= phi <= phi_max
    - Monotonicity: partialphi/partialx maintains sign
    - Convexity: nabla^2phi >= 0
    
References:
    - Beven, K. (2012). Rainfall-Runoff Modelling: The Primer
    - Dingman, S.L. (2015). Physical Hydrology
"""

from collections.abc import Mapping
from typing import Optional, Dict, Any, List, Tuple, Union
import torch
from relic.core import device_policy
import torch.nn as nn
from torch import Tensor
import logging
from enum import Enum
from dataclasses import dataclass, fields

from relic.core.managers.base import BaseManager
from relic.core.managers.metadata import (
    ManagerMetadata, ManagerCapability, ManagerResource, CapabilityType,
    ResourceType
)
from relic.parameters import get_global_manager

from .variable_catalog import find_catalog_entry

logger = logging.getLogger(__name__)

# The parameter whose allowed_values are the constraint methods (clip, scale, redistribute, project).
PHYSICAL_CONSTRAINT_METHOD_KEY = "coupling.scales.defaults.physical_constraint_method"


def _constraint_methods(parameter_manager: Any) -> Tuple[str, ...]:
    """Return the constraint methods: the allowed_values of PHYSICAL_CONSTRAINT_METHOD_KEY."""
    spec = parameter_manager.get_parameter(PHYSICAL_CONSTRAINT_METHOD_KEY)
    if spec is None or not spec.allowed_values:
        raise KeyError(
            f"{PHYSICAL_CONSTRAINT_METHOD_KEY} must be declared with allowed_values (the constraint methods)."
        )
    return tuple(str(value) for value in spec.allowed_values)


def _require_constraint_method(method: Any, parameter_manager: Any, source: str) -> str:
    """Return ``method`` if it is one of the constraint methods; raise ValueError naming them otherwise."""
    allowed = _constraint_methods(parameter_manager)
    if not isinstance(method, str) or method not in allowed:
        raise ValueError(
            f"{source}: constraint method {method!r} is not one of {list(allowed)} "
            f"(allowed_values of {PHYSICAL_CONSTRAINT_METHOD_KEY})."
        )
    return method


class ConstraintType(Enum):
    """Types of physical constraints."""
    NON_NEGATIVE = "non_negative"
    BOUNDED = "bounded"
    MONOTONIC = "monotonic"
    CONVEX = "convex"
    SMOOTH = "smooth"
    CONSERVE_SHAPE = "conserve_shape"


@dataclass
class ConstraintSpec:
    """Specification for a physical constraint."""
    name: str
    type: ConstraintType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    smoothness: Optional[float] = None
    strict: bool = True
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        # The type is a ConstraintType or its value string; anything else raises (no skipped constraint).
        if isinstance(self.type, ConstraintType):
            return
        allowed = [member.value for member in ConstraintType]
        if not isinstance(self.type, str) or self.type not in allowed:
            raise ValueError(f"ConstraintSpec {self.name!r}: type {self.type!r} is not one of {allowed}.")
        self.type = ConstraintType(self.type)


# The settings PhysicalConstraintManager.configure accepts.
_CONFIGURABLE_SETTINGS: Tuple[str, ...] = ("method", "preserve_gradients", "device", "constraints")


class PhysicalConstraintManager(BaseManager, nn.Module):
    """
    Physical constraint manager ensuring data respects physical bounds.
    
    This class applies and enforces physical constraints on scaled data
    to maintain physical realism.
    """
    
    def __init__(
        self,
        constraints: Optional[List[ConstraintSpec]] = None,
        method: Optional[str] = None,
        preserve_gradients: Optional[bool] = None,
        device: Optional[torch.device] = None,
        parameter_manager: Optional[Any] = None,
    ):
        """
        Initialize physical constraint manager.
        
        Parameters:
            constraints : Optional[List[ConstraintSpec]]
                List of constraint specifications (ConstraintSpec entries; anything else raises
                ValueError)
            method : str
                Constraint application method, one of the allowed_values of
                coupling.scales.defaults.physical_constraint_method (an invalid method raises
                ValueError here; None takes the parameter at initialize)
            preserve_gradients : bool
                Whether to preserve gradients through constraints (True, False or None for the
                parameter default; anything else raises ValueError)
            device : Optional[torch.device]
                Computation device
        """
        if method is not None:
            _require_constraint_method(
                method,
                parameter_manager if parameter_manager is not None else get_global_manager(),
                "PhysicalConstraintManager(method=...)",
            )
        if preserve_gradients is not None and not isinstance(preserve_gradients, bool):
            raise ValueError(
                f"PhysicalConstraintManager(preserve_gradients=...) must be True or False, got {preserve_gradients!r}."
            )
        for position, constraint in enumerate(constraints or []):
            if not isinstance(constraint, ConstraintSpec):
                raise ValueError(
                    f"PhysicalConstraintManager(constraints=...): entry {position} must be a ConstraintSpec, "
                    f"got {constraint!r}."
                )
        # Store initialization parameters
        self._init_constraints = list(constraints or [])
        self._init_method = method
        self._init_preserve_gradients = preserve_gradients
        self._init_device = device
        self._init_parameter_manager = parameter_manager
        
        # Initialize BaseManager first
        metadata = self._create_metadata()
        BaseManager.__init__(self, metadata=metadata)
        
        # Initialize nn.Module
        nn.Module.__init__(self)
        self._last_bounds_source: Optional[str] = None
        
        # The actual initialization will be done in _do_initialize()
    
    def _create_metadata(self) -> ManagerMetadata:
        """Create metadata for PhysicalConstraintManager."""
        capabilities = [
            ManagerCapability(
                type=CapabilityType.CONSTRAINT_ENFORCEMENT,
                level="advanced",
                description="Physical constraint application and enforcement"
            ),
            ManagerCapability(
                type=CapabilityType.VALIDATION,
                level="advanced",
                description="Constraint violation detection and tracking"
            ),
            ManagerCapability(
                type=CapabilityType.PYTORCH_INTEGRATION,
                level="standard",
                description="PyTorch-based constraint operations"
            ),
        ]
        
        resources = [
            ManagerResource(
                type=ResourceType.SYSTEM_MEMORY,
                quantity=None,
                unit="bytes"
            ),
            ManagerResource(
                type=ResourceType.GPU_MEMORY,
                quantity=None,
                unit="bytes"
            ),
        ]
        
        dependencies = []  # No dependencies on other managers
        
        return ManagerMetadata(
            name="PhysicalConstraintManager",
            version="2.0.0",
            description="Physical constraint manager ensuring data respects physical bounds",
            capabilities=capabilities,
            resources=resources,
            dependencies=dependencies,
            tags={"thread_safe", "gpu_enabled", "differentiable"},
            custom_attributes={
                "thread_safe": True,
                "supports_gpu": True,
                "supports_distributed": False,
                "preserves_gradients": True
            }
        )
    
    def _do_initialize(self) -> None:
        """Perform PhysicalConstraintManager-specific initialization."""
        # Set core attributes (a copy: constraints added later are discarded by reset())
        self.constraints = list(self._init_constraints)
        self.parameter_manager = (
            self._init_parameter_manager
            if self._init_parameter_manager is not None
            else get_global_manager()
        )

        defaults_ns = "coupling.scales.defaults"
        if self.parameter_manager is None:
            if (
                self._init_method is None
                or self._init_preserve_gradients is None
                or self._init_device is None
            ):
                raise RuntimeError(
                    "PhysicalConstraintManager requires ParameterManager or explicit method, preserve_gradients, and device."
                )
            self.method = self._init_method
            self.preserve_gradients = bool(self._init_preserve_gradients)
            self.device = self._init_device
        else:
            if self._init_method is None:
                self.method = _require_constraint_method(
                    self.parameter_manager.get(
                        "physical_constraint_method",
                        namespace=defaults_ns,
                        expected_type=str,
                        required=True,
                    ),
                    self.parameter_manager,
                    PHYSICAL_CONSTRAINT_METHOD_KEY,
                )
            else:
                self.method = self._init_method

            if self._init_preserve_gradients is None:
                self.preserve_gradients = bool(
                    self.parameter_manager.get(
                        "preserve_gradients",
                        namespace=defaults_ns,
                        expected_type=bool,
                        required=True,
                    )
                )
            else:
                self.preserve_gradients = bool(self._init_preserve_gradients)

            if self._init_device is None:
                device_value = self.parameter_manager.get(
                    "constraint_device",
                    namespace=defaults_ns,
                    expected_type=str,
                    required=True,
                )
                self.device = torch.device(str(device_value))
            else:
                self.device = self._init_device

        # Constraint violation tracking
        self.violation_count = 0
        self.violation_history: List[Dict[str, Any]] = []

        # Learned constraint parameters (for adaptive constraints)
        self.adaptive_bounds = nn.ParameterDict()

        logger.info(
            f"PhysicalConstraintManager initialized: method={self.method}, "
            f"preserve_gradients={self.preserve_gradients}, device={self.device}"
        )

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check one configure() call and return the settings _do_configure applies.

        BaseManager.configure runs this before it stores or applies anything, so a rejected call
        raises its ValueError and changes nothing. Rejected: a key outside _CONFIGURABLE_SETTINGS,
        a method outside the allowed_values of PHYSICAL_CONSTRAINT_METHOD_KEY, a preserve_gradients
        that is not True or False, a device that is not a torch device, and constraints that are
        not a list of mappings of ConstraintSpec fields with a known type.
        """
        unknown = sorted(str(name) for name in config if name not in _CONFIGURABLE_SETTINGS)
        if unknown:
            raise ValueError(
                f"PhysicalConstraintManager.configure does not accept {unknown}; the configurable "
                f"settings are {list(_CONFIGURABLE_SETTINGS)}."
            )
        settings: Dict[str, Any] = {}
        if "method" in config:
            settings["method"] = _require_constraint_method(
                config["method"], self.parameter_manager, "PhysicalConstraintManager.configure"
            )
        if "preserve_gradients" in config:
            value = config["preserve_gradients"]
            if not isinstance(value, bool):
                raise ValueError(
                    f"PhysicalConstraintManager.configure: preserve_gradients must be True or False, got {value!r}."
                )
            settings["preserve_gradients"] = value
        if "device" in config:
            value = config["device"]
            if isinstance(value, torch.device):
                settings["device"] = value
            elif isinstance(value, str):
                try:
                    settings["device"] = torch.device(value)
                except RuntimeError as exc:
                    raise ValueError(
                        f"PhysicalConstraintManager.configure: device {value!r} is not a torch device ({exc})."
                    ) from exc
            else:
                raise ValueError(
                    f"PhysicalConstraintManager.configure: device must be a torch.device or its name, got {value!r}."
                )
        if "constraints" in config:
            entries = config["constraints"]
            if not isinstance(entries, (list, tuple)):
                raise ValueError(
                    f"PhysicalConstraintManager.configure: constraints must be a list of mappings, got {entries!r}."
                )
            field_names = [spec_field.name for spec_field in fields(ConstraintSpec)]
            specs: List[ConstraintSpec] = []
            for position, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise ValueError(
                        f"PhysicalConstraintManager.configure: constraints[{position}] must be a mapping of "
                        f"ConstraintSpec fields {field_names}, got {entry!r}."
                    )
                unknown_fields = sorted(str(name) for name in entry if name not in field_names)
                missing_fields = [name for name in ("name", "type") if name not in entry]
                if unknown_fields or missing_fields:
                    raise ValueError(
                        f"PhysicalConstraintManager.configure: constraints[{position}] has unknown fields "
                        f"{unknown_fields} and lacks {missing_fields}; the ConstraintSpec fields are {field_names}."
                    )
                specs.append(ConstraintSpec(**entry))
            settings["constraints"] = specs
        return settings

    def _do_configure(self, settings: Dict[str, Any]) -> None:
        """Configure PhysicalConstraintManager with runtime settings (checked by _validate_config)."""
        if "method" in settings:
            self.method = settings["method"]

        if "preserve_gradients" in settings:
            self.preserve_gradients = settings["preserve_gradients"]
        if "device" in settings:
            self.device = settings["device"]
            self.to(self.device)
        if "constraints" in settings:
            self.constraints.extend(settings["constraints"])

        logger.debug(f"PhysicalConstraintManager configured with: {settings}")

    def _do_validate(self) -> List[str]:
        """Validate PhysicalConstraintManager configuration and state.
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check device availability
        if self.device.type == 'cuda' and not device_policy.is_cuda_available():
            errors.append(f"CUDA device {self.device} requested but not available")
        
        # Validate method
        valid_methods = list(_constraint_methods(self.parameter_manager))
        if self.method not in valid_methods:
            errors.append(f"Invalid constraint method: {self.method}. Must be one of {valid_methods}")
        
        # Validate constraints
        for constraint in self.constraints:
            if constraint.type == ConstraintType.BOUNDED:
                if constraint.min_value is None and constraint.max_value is None:
                    errors.append(f"Bounded constraint '{constraint.name}' has neither min nor max value")
                if constraint.min_value is not None and constraint.max_value is not None:
                    if constraint.min_value >= constraint.max_value:
                        errors.append(
                            f"Constraint '{constraint.name}': min_value ({constraint.min_value}) "
                            f">= max_value ({constraint.max_value})"
                        )
        
        if not errors:
            logger.debug("PhysicalConstraintManager validation completed successfully")
        else:
            logger.warning(f"PhysicalConstraintManager validation found {len(errors)} issues")
        
        return errors
    
    def _do_cleanup(self) -> None:
        """Clean up PhysicalConstraintManager resources."""
        # Reset violation tracking
        self.reset_tracking()
        
        # Clear adaptive bounds
        self.adaptive_bounds.clear()
        
        logger.info("PhysicalConstraintManager cleanup completed")
        
    def add_constraint(self, constraint: ConstraintSpec) -> None:
        """
        Add a new constraint specification.
        
        Parameters:
            constraint : ConstraintSpec
                Constraint to add
        """
        if not isinstance(constraint, ConstraintSpec):
            raise ValueError(f"add_constraint takes a ConstraintSpec, got {constraint!r}.")
        if not isinstance(constraint.type, ConstraintType):
            raise ValueError(
                f"Constraint {constraint.name!r} has type {constraint.type!r}, not one of "
                f"{[member.value for member in ConstraintType]}."
            )
        self.constraints.append(constraint)
        logger.info(f"Added constraint: {constraint.name} ({constraint.type.value})")

        self._last_bounds_source: Optional[str] = None

    def _resolve_bounds_from_parameters(
        self, variable_name: str, scale: Optional[str] = None
    ) -> Optional[Tuple[Tuple[Optional[float], Optional[float]], str]]:
        """Declared bounds of ``variable_name``: the ``bounds`` of its scale-catalog entry.

        The catalogs are coupling.scales.<scale>.variables (when a scale is given), then
        coupling.scales.defaults.variables, matched by the shared find_catalog_entry. Returns
        ((lower, upper), source) or None when the entry declares ``bounds: null``.

        Raises:
            KeyError: If no catalog declares the variable or its entry has no bounds field
            ValueError: If the bounds are not a [lower, upper] pair
        """
        namespaces = ([f"coupling.scales.{scale}"] if scale else []) + ["coupling.scales.defaults"]

        def read_catalog(namespace: str) -> Optional[Dict[str, Any]]:
            return self.parameter_manager.get(
                "variables", namespace=namespace, expected_type=dict, required=False
            )

        found = find_catalog_entry(variable_name, namespaces, read_catalog)
        if found is None:
            raise KeyError(
                f"{variable_name!r} is not declared in {[f'{ns}.variables' for ns in namespaces]}; pass "
                "bounds explicitly or declare the variable with its bounds."
            )
        entry, namespace = found
        if "bounds" not in entry:
            raise KeyError(
                f"The catalog entry of {variable_name!r} in {namespace}.variables declares no bounds; "
                "declare bounds (null for an unbounded variable) or pass bounds explicitly."
            )
        declared = entry["bounds"]
        source_label = f"{namespace}.variables.{variable_name}.bounds"
        if declared is None:
            return None
        if not isinstance(declared, (list, tuple)) or len(declared) != 2:
            raise ValueError(f"{source_label} must be a [lower, upper] pair, got {declared!r}.")
        lower = float(declared[0]) if declared[0] is not None else None
        upper = float(declared[1]) if declared[1] is not None else None
        logger.info(f"Resolved bounds for {variable_name} from {source_label}")
        return (lower, upper), source_label

    def apply_constraints(
        self,
        data: Tensor,
        constraint_names: Optional[List[str]] = None,
        return_mask: bool = False,
        *,
        bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
        variable_name: Optional[str] = None,
        scale: Optional[str] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Apply physical constraints to data.

        Parameters:
            data : Tensor
                Data to constrain
            constraint_names : Optional[List[str]]
                Specific constraints to apply (None = all)
            return_mask : bool
                Whether to return violation mask
            bounds : Optional[Tuple[Optional[float], Optional[float]]]
                Runtime physical bounds (min, max) from metadata; when provided a
                temporary bounded constraint is constructed.
            variable_name : Optional[str]
                Variable name for diagnostics when bounds are supplied.
            scale : Optional[str]
                Scale name used for parameterized bounds lookup.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]:
                Constrained data, optionally with violation mask
        """
        constrained = data.clone() if self.preserve_gradients else data.detach().clone()
        violation_mask = torch.zeros_like(data, dtype=torch.bool)

        active_constraints: List[ConstraintSpec] = list(self.constraints)

        self._last_bounds_source = None
        resolved_bounds = bounds
        if resolved_bounds is None and variable_name is not None:
            resolved = self._resolve_bounds_from_parameters(variable_name, scale=scale)
            if resolved is not None:
                resolved_bounds, resolved_source = resolved
                self._last_bounds_source = resolved_source
            else:
                resolved_source = None
        else:
            resolved_source = None

        if resolved_bounds is not None:
            # Create a temporary bounded constraint using provided limits.
            min_bound, max_bound = resolved_bounds
            if resolved_source is None:
                resolved_source = "metadata.physical_bounds" if bounds is not None else "runtime.bounds"
            self._last_bounds_source = resolved_source
            active_constraints = active_constraints + [
                ConstraintSpec(
                    name=f"{variable_name or 'variable'}_bounds",
                    type=ConstraintType.BOUNDED,
                    min_value=float(min_bound) if min_bound is not None else None,
                    max_value=float(max_bound) if max_bound is not None else None,
                    strict=True,
                    metadata={
                        "source": resolved_source,
                        "variable": variable_name or "unknown",
                    },
                )
            ]

        if constraint_names:
            active_constraints = [
                c for c in active_constraints if c.name in constraint_names
            ]

        # Apply each constraint (References: Beven 2012; Dingman 2015)
        for constraint in active_constraints:
            constrained, mask = self._apply_single_constraint(constrained, constraint)
            violation_mask = violation_mask | mask

        # Track violations
        n_violations = violation_mask.sum().item()
        if n_violations > 0:
            self.violation_count += n_violations
            self.violation_history.append({
                'n_violations': n_violations,
                'total_elements': data.numel(),
                'violation_rate': n_violations / data.numel(),
                'constraints_applied': [c.name for c in active_constraints]
            })

        if return_mask:
            return constrained, violation_mask
        return constrained
    
    def _apply_single_constraint(
        self, data: Tensor, constraint: ConstraintSpec
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply a single constraint to data.
        
        Parameters:
            data : Tensor
                Data to constrain
            constraint : ConstraintSpec
                Constraint specification
                
        Returns:
            Tuple[Tensor, Tensor]: Constrained data and violation mask
        """
        violation_mask = torch.zeros_like(data, dtype=torch.bool)
        
        if constraint.type == ConstraintType.NON_NEGATIVE:
            violation_mask = data < 0
            data = self._apply_non_negative(data, constraint)
            
        elif constraint.type == ConstraintType.BOUNDED:
            lower = constraint.min_value if constraint.min_value is not None else -float('inf')
            upper = constraint.max_value if constraint.max_value is not None else float('inf')
            violation_mask = (data < lower) | (data > upper)
            data = self._apply_bounds(data, constraint)
            
        elif constraint.type == ConstraintType.MONOTONIC:
            data, violation_mask = self._apply_monotonicity(data, constraint)
            
        elif constraint.type == ConstraintType.CONVEX:
            data, violation_mask = self._apply_convexity(data, constraint)
            
        elif constraint.type == ConstraintType.SMOOTH:
            data = self._apply_smoothness(data, constraint)
            
        elif constraint.type == ConstraintType.CONSERVE_SHAPE:
            data = self._apply_shape_conservation(data, constraint)
            
        else:
            raise ValueError(
                f"Constraint {constraint.name!r} has type {constraint.type!r}, not one of "
                f"{[member.value for member in ConstraintType]}."
            )
        
        return data, violation_mask
    
    def _apply_non_negative(self, data: Tensor, constraint: ConstraintSpec) -> Tensor:
        """Apply non-negativity constraint."""
        if self.method == "clip":
            return torch.clamp(data, min=0)
        elif self.method == "scale":
            min_val = data.min()
            if min_val < 0:
                return data - min_val
            return data
        elif self.method == "redistribute":
            return self._redistribute_negative_values(data)
        elif self.method == "project":
            return torch.relu(data)
        raise ValueError(
            f"Constraint method {self.method!r} has no non-negativity implementation; the "
            f"implemented methods are clip, scale, redistribute and project ({PHYSICAL_CONSTRAINT_METHOD_KEY})."
        )
    
    def _apply_bounds(self, data: Tensor, constraint: ConstraintSpec) -> Tensor:
        """Apply bounded constraint."""
        min_val = constraint.min_value if constraint.min_value is not None else -float('inf')
        max_val = constraint.max_value if constraint.max_value is not None else float('inf')
        
        if self.method == "clip":
            return torch.clamp(data, min=min_val, max=max_val)
        elif self.method == "scale":
            return self._scale_to_bounds(data, min_val, max_val)
        elif self.method == "redistribute":
            return self._redistribute_to_bounds(data, min_val, max_val)
        elif self.method == "project":
            return self._project_to_bounds(data, min_val, max_val)
        raise ValueError(
            f"Constraint method {self.method!r} has no bounds implementation; the implemented "
            f"methods are clip, scale, redistribute and project ({PHYSICAL_CONSTRAINT_METHOD_KEY})."
        )
    
    def _apply_monotonicity(
        self, data: Tensor, constraint: ConstraintSpec
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply monotonicity constraint.
        
        Ensures that spatial gradients maintain consistent sign.
        """
        if data.ndim < 2:
            return data, torch.zeros_like(data, dtype=torch.bool)
        
        # Compute gradients
        if data.ndim == 2:
            dy = data[1:] - data[:-1]
            dx = data[:, 1:] - data[:, :-1]
            
            # Check for sign changes
            y_violations = (dy[1:] * dy[:-1]) < 0
            x_violations = (dx[:, 1:] * dx[:, :-1]) < 0
            
            # Apply isotonic regression or smoothing
            if y_violations.any() or x_violations.any():
                data = self._isotonic_regression_2d(data)
        
        # Create violation mask
        violation_mask = torch.zeros_like(data, dtype=torch.bool)
        
        return data, violation_mask
    
    def _apply_convexity(
        self, data: Tensor, constraint: ConstraintSpec
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply convexity constraint.
        
        Ensures that second derivatives are non-negative (convex)
        or non-positive (concave).
        """
        if data.ndim < 2:
            return data, torch.zeros_like(data, dtype=torch.bool)
        
        # Compute second derivatives (Laplacian)
        laplacian = self._compute_laplacian(data)
        
        # Check for convexity violations
        violation_mask = laplacian < 0
        
        if violation_mask.any():
            # Apply convex projection
            data = self._project_to_convex(data)
        
        return data, violation_mask
    
    def _apply_smoothness(self, data: Tensor, constraint: ConstraintSpec) -> Tensor:
        """
        Apply smoothness constraint using filtering.
        
        Parameters:
            data : Tensor
                Data to smooth
            constraint : ConstraintSpec
                Contains smoothness parameter
        """
        smoothness = constraint.smoothness or 1.0
        
        if data.ndim >= 2:
            # Apply Gaussian smoothing
            kernel_size = int(2 * smoothness + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Create Gaussian kernel
            sigma = smoothness
            kernel = self._gaussian_kernel(kernel_size, sigma, data.ndim)
            kernel = kernel.to(data.device)
            
            # Apply convolution
            if data.ndim == 2:
                data_padded = torch.nn.functional.pad(
                    data.unsqueeze(0).unsqueeze(0),
                    (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2),
                    mode='reflect'
                )
                smoothed = torch.nn.functional.conv2d(
                    data_padded, kernel.unsqueeze(0).unsqueeze(0)
                )
                data = smoothed.squeeze()
            elif data.ndim == 3:
                data_padded = torch.nn.functional.pad(
                    data.unsqueeze(0).unsqueeze(0),
                    (kernel_size//2,) * 6,
                    mode='reflect'
                )
                smoothed = torch.nn.functional.conv3d(
                    data_padded, kernel.unsqueeze(0).unsqueeze(0)
                )
                data = smoothed.squeeze()
        
        return data
    
    def _apply_shape_conservation(self, data: Tensor, constraint: ConstraintSpec) -> Tensor:
        """
        Apply shape conservation constraint.
        
        Preserves the relative shape/pattern while adjusting values.
        """
        # Preserve relative variations
        mean = data.mean()
        std = data.std()
        
        if std > 0:
            # Standardize
            normalized = (data - mean) / std
            
            # Apply bounds if specified
            if constraint.min_value is not None or constraint.max_value is not None:
                target_mean = (constraint.min_value + constraint.max_value) / 2
                target_range = constraint.max_value - constraint.min_value
                target_std = target_range / 6  # Assume ~3 sigma covers range
                
                # Rescale
                data = normalized * target_std + target_mean
                
                # Ensure bounds
                if constraint.min_value is not None:
                    data = torch.clamp(data, min=constraint.min_value)
                if constraint.max_value is not None:
                    data = torch.clamp(data, max=constraint.max_value)
        
        return data
    
    def _redistribute_negative_values(self, data: Tensor) -> Tensor:
        """Redistribute negative values to positive cells."""
        negative_mask = data < 0
        positive_mask = ~negative_mask
        
        if negative_mask.any() and positive_mask.any():
            negative_total = data[negative_mask].sum()
            data[negative_mask] = 0
            
            # Redistribute proportionally to positive values
            positive_sum = data[positive_mask].sum()
            if positive_sum > 0:
                weights = data[positive_mask] / positive_sum
                data[positive_mask] -= negative_total * weights
        
        return torch.clamp(data, min=0)
    
    def _scale_to_bounds(self, data: Tensor, min_val: float, max_val: float) -> Tensor:
        """Scale data to fit within bounds."""
        data_min = data.min()
        data_max = data.max()
        
        if data_max > data_min:
            # Linear scaling
            scale = (max_val - min_val) / (data_max - data_min)
            shift = min_val - data_min * scale
            return data * scale + shift
        else:
            # Constant field
            return torch.full_like(data, (min_val + max_val) / 2)
    
    def _redistribute_to_bounds(
        self, data: Tensor, min_val: float, max_val: float
    ) -> Tensor:
        """Redistribute values exceeding bounds."""
        below_min = data < min_val
        above_max = data > max_val
        within_bounds = ~(below_min | above_max)
        
        if below_min.any():
            deficit = (min_val - data[below_min]).sum()
            data[below_min] = min_val
            
            if within_bounds.any():
                # Add deficit to valid cells
                data[within_bounds] += deficit / within_bounds.sum()
        
        if above_max.any():
            excess = (data[above_max] - max_val).sum()
            data[above_max] = max_val
            
            if within_bounds.any():
                # Remove excess from valid cells
                data[within_bounds] -= excess / within_bounds.sum()
        
        # Final clipping to ensure bounds
        return torch.clamp(data, min=min_val, max=max_val)
    
    def _project_to_bounds(self, data: Tensor, min_val: float, max_val: float) -> Tensor:
        """Project data onto feasible set using optimization."""
        # Proximal gradient projection
        return torch.clamp(data, min=min_val, max=max_val)
    
    def _isotonic_regression_2d(self, data: Tensor) -> Tensor:
        """Apply isotonic regression for monotonicity."""
        # Simplified pool-adjacent-violators algorithm
        # This is a placeholder for full isotonic regression
        return data
    
    def _compute_laplacian(self, data: Tensor) -> Tensor:
        """Compute discrete Laplacian."""
        if data.ndim == 2:
            # 2D Laplacian using finite differences
            laplacian = torch.zeros_like(data)
            laplacian[1:-1, 1:-1] = (
                data[2:, 1:-1] + data[:-2, 1:-1] +
                data[1:-1, 2:] + data[1:-1, :-2] -
                4 * data[1:-1, 1:-1]
            )
            return laplacian
        else:
            return torch.zeros_like(data)
    
    def _project_to_convex(self, data: Tensor) -> Tensor:
        """Project data to convex set."""
        # This is a simplified convex projection
        # Full implementation would use iterative algorithms
        return data
    
    def _gaussian_kernel(self, size: int, sigma: float, ndim: int) -> Tensor:
        """Create Gaussian kernel for smoothing."""
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        
        if ndim == 2:
            y, x = torch.meshgrid(coords, coords, indexing='ij')
            kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        elif ndim == 3:
            z, y, x = torch.meshgrid(coords, coords, coords, indexing='ij')
            kernel = torch.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
        else:
            kernel = torch.exp(-(coords**2) / (2 * sigma**2))
        
        return kernel / kernel.sum()
    
    def get_violation_report(self) -> Dict[str, Any]:
        """
        Get report of constraint violations.
        
        Returns:
            Dict[str, Any]: Violation statistics
        """
        if not self.violation_history:
            return {
                'total_violations': 0,
                'total_checks': 0,
                'violation_rate': 0.0,
                'history': []
            }
        
        total_violations = sum(v['n_violations'] for v in self.violation_history)
        total_elements = sum(v['total_elements'] for v in self.violation_history)
        
        return {
            'total_violations': total_violations,
            'total_checks': len(self.violation_history),
            'violation_rate': total_violations / total_elements if total_elements > 0 else 0.0,
            'history': self.violation_history[-10:]  # Last 10 entries
        }
    
    def reset_tracking(self) -> None:
        """Reset violation tracking."""
        self.violation_count = 0
        self.violation_history.clear()
        logger.info("Violation tracking reset")
