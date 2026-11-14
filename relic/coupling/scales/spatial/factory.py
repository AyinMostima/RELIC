"""
Spatial Scale Coupling Factory Module

Provides automatic selection of spatial regridding methods based on variable type,
conservation requirements, and performance considerations.

Mathematical Expression:
    Method selection function: M = f(variable_type, conservation_required, grid_consistency)
    
    Decision tree:
    - Extensive variables (fluxes) ? Conservative remapping
    - Intensive variables (states) ? Bilinear interpolation
    - Categorical variables ? Nearest neighbor
    - Override with explicit method specification

References:
    - Jones, P.W. (1999). First- and second-order conservative remapping schemes for grids in spherical coordinates
    - Ramshaw, J.D. (1985). Conservative rezoning algorithm for generalized two-dimensional meshes
"""

import torch
from relic.core import device_policy
from torch import Tensor
from typing import Dict, Optional, Any, Literal, Tuple, List
import logging

from .regrid_conservative import ConservativeRegridder
from .regrid_bilinear import BilinearRegridder
from .regrid_nearest import NearestNeighborRegridder
from .grid_utils import GridDescriptor
from ....core.abstractions import ScaleCoupling
from ....core.functor import (
    ScaleCouplingFunctor,
    FunctorMetadata,
    ProcessCategory,
    ExecutionPhase,
    SpatialScale,
    VariableKind,
)
from relic.parameters import get_global_manager
from ...core.conservation_checker import require_relative_tolerance
from ...core.policy import ScalingMode
from .grid_utils import content_fingerprint
from ..types import SpatialQuantityBasis, resolve_spatial_basis
logger = logging.getLogger(__name__)

NEAREST_METHODS = ('nearest', 'irregular_nearest')


def declared_regrid_basis(
    variable_metadata: Optional[Dict[str, Any]],
    quantity_basis: Optional[Any] = None,
) -> Optional[SpatialQuantityBasis]:
    """The declared SpatialQuantityBasis of a regridded field (explicit argument, else metadata), or None."""
    if quantity_basis is not None:
        return SpatialQuantityBasis.parse(quantity_basis)
    if isinstance(variable_metadata, dict) and (
        'spatial_basis' in variable_metadata or 'variable_type' in variable_metadata
    ):
        return resolve_spatial_basis('regridded field', variable_metadata)
    return None


def check_regrid_method_for_basis(
    method: str,
    basis: Optional[SpatialQuantityBasis],
    enforce_conservation: bool,
) -> None:
    """
    The one basis-aware rule for the spatial regridding method.

    A categorical field is regridded by nearest neighbour only (class codes have no arithmetic,
    so conservation does not apply to them). Under enforce_conservation every other field -- an
    amount (cell_total, area_density) or an intensive field -- is regridded by the conservative
    method; a non-conservative method needs the declared basis to be recognised as the
    categorical exemption. Violations raise.
    """
    if basis is SpatialQuantityBasis.CATEGORICAL:
        if method not in NEAREST_METHODS:
            raise ValueError(
                f"A categorical field is regridded by nearest neighbour only; method {method!r} would "
                "average or split class codes."
            )
        return
    if not enforce_conservation or method == 'conservative':
        return
    if basis is None:
        raise ValueError(
            f"Explicit method {method!r} conflicts with enforce_conservation=True unless the field is "
            "declared categorical; no spatial basis is declared (spatial_basis)."
        )
    raise ValueError(
        f"Explicit method {method!r} conflicts with enforce_conservation=True for a {basis.value} field; "
        "amounts and intensive fields are regridded conservatively under enforcement."
    )


class SpatialRegridderFactory:
    """
    Factory for explicitly selected spatial regridding algorithms.

    Standard names, units, and broad variable roles are insufficient to infer
    a scientifically valid numerical regridding method.
    """
    
    # Method classes
    REGRIDDER_CLASSES = {
        'conservative': ConservativeRegridder,
        'bilinear': BilinearRegridder,
        'nearest': NearestNeighborRegridder,
        'irregular_bilinear': BilinearRegridder,
        'irregular_nearest': NearestNeighborRegridder
    }
    
    def __init__(
        self,
        enforce_conservation: bool = False,
        conservation_tolerance: Optional[float] = None,
        device: Optional[torch.device] = None,
        parameter_manager: Optional[Any] = None
    ):
        """
        Initialize spatial regridder factory.
        
        Parameters:
            enforce_conservation : bool
                If True, force conservative remapping for all variables
            conservation_tolerance : Optional[float]
                Explicit relative tolerance of the conservative regridders of this factory
                (finite, in (0, 1)); None keeps the one regridder default,
                coupling.scales.defaults.conservation_tolerance, read by each regridder
                (SpatialRegridderBase). A per-call conservation_tolerance of create_regridder
                replaces it for that regridder.
        device : Optional[torch.device]
            Computation device
        """
        self.enforce_conservation = enforce_conservation

        self.parameter_manager = parameter_manager or get_global_manager()

        # The factory resolves no tolerance of its own: an explicit value, or None for the one
        # regridder default (no second default such as validation.conservation).
        self.conservation_tolerance: Optional[float] = (
            None
            if conservation_tolerance is None
            else require_relative_tolerance(conservation_tolerance, "conservation_tolerance")
        )
        self.device = device or (
            device_policy.resolve_device(default="auto")
        )
        
        # Cache for regridder instances
        self._regridder_cache: Dict[str, ScaleCoupling] = {}

    def create_regridder(
        self,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        method: Optional[str] = None,
        variable_metadata: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ScaleCoupling:
        """
        Create one spatial regridder from an explicit method contract.
        
        Parameters:
            source_grid : GridDescriptor
                Source grid description
            target_grid : GridDescriptor
                Target grid description
            method : Optional[str]
                Explicit method ('conservative', 'bilinear', 'nearest')
            variable_metadata : Optional[Dict[str, Any]]
                Metadata containing an explicit ``regridding_method`` when
                ``method`` is not supplied
            **kwargs : Dict
                Additional arguments for regridder initialization
                
        Returns:
            ScaleCoupling: Appropriate regridder instance
            
        Raises:
            ValueError: If method cannot be determined or is invalid
        """
        # Support class-style usage: SpatialRegridderFactory.create_regridder("bilinear", src, tgt, ...)
        if isinstance(self, (str, ScalingMode)) or not isinstance(self, SpatialRegridderFactory):
            method_override = self if isinstance(self, (str, ScalingMode)) else source_grid
            src_grid = source_grid if isinstance(self, (str, ScalingMode)) else target_grid
            tgt_grid = target_grid if isinstance(self, (str, ScalingMode)) else method
            if not isinstance(method_override, (str, ScalingMode)):
                raise TypeError(
                    "Class-level usage must be SpatialRegridderFactory.create_regridder(method, source_grid, target_grid, ...)"
                )
            if not isinstance(src_grid, GridDescriptor) or not isinstance(tgt_grid, GridDescriptor):
                raise TypeError(
                    "Class-level usage must be SpatialRegridderFactory.create_regridder(method, source_grid, target_grid, ...)"
                )
            factory_kwargs = {}
            for key in (
                "enforce_conservation",
                "conservation_tolerance",
                "device",
                "parameter_manager",
            ):
                if key in kwargs:
                    factory_kwargs[key] = kwargs.pop(key)
            factory = SpatialRegridderFactory(**factory_kwargs)
            return factory.create_regridder(
                src_grid,
                tgt_grid,
                method=str(method_override),
                variable_metadata=variable_metadata,
                **kwargs,
            )
        factory = self
        # Determine method
        if method is None:
            if variable_metadata is None:
                raise ValueError(
                    "An explicit regridding_method is required; variable "
                    "metadata and missing values cannot select an algorithm"
                )
            method = self._select_method(variable_metadata)
        
        # Validate method
        if method not in self.REGRIDDER_CLASSES:
            raise ValueError(
                f"Invalid regridding method: {method}. "
                f"Must be one of {list(self.REGRIDDER_CLASSES.keys())}"
            )

        # One basis-aware rule: categorical fields use nearest neighbour; under enforcement (factory
        # policy or the variable's require_conservation) every other field is remapped conservatively.
        check_regrid_method_for_basis(
            method,
            declared_regrid_basis(variable_metadata, kwargs.get('quantity_basis')),
            factory.enforce_conservation
            or bool(isinstance(variable_metadata, dict) and variable_metadata.get('require_conservation')),
        )

        # The conservative kernel needs one explicit SpatialQuantityBasis: an explicit
        # quantity_basis argument, else the spatial_basis/variable_type of variable_metadata
        # (a bare "extensive" raises as ambiguous). Without either, the regridder is built
        # undeclared and each remap must pass its own quantity_basis.
        if method == 'conservative':
            quantity_basis = kwargs.pop('quantity_basis', None)
            if quantity_basis is None and isinstance(variable_metadata, dict) and (
                'spatial_basis' in variable_metadata or 'variable_type' in variable_metadata
            ):
                quantity_basis = resolve_spatial_basis('regridded field', variable_metadata)
            if quantity_basis is not None:
                kwargs['quantity_basis'] = SpatialQuantityBasis.parse(quantity_basis)
            # Tolerance of this regridder: the per-call value (the ScaleManager facade passes the
            # ledger tolerance of the call's mode), else the factory's explicit value, else none
            # here, so that SpatialRegridderBase reads the one default
            # coupling.scales.defaults.conservation_tolerance.
            tolerance = kwargs.pop('conservation_tolerance', None)
            if tolerance is None:
                tolerance = factory.conservation_tolerance
            if tolerance is not None:
                kwargs['conservation_tolerance'] = require_relative_tolerance(tolerance, "conservation_tolerance")
        else:
            for argument in ('quantity_basis', 'conservation_tolerance'):
                if argument in kwargs:
                    raise ValueError(f"{argument} applies to the conservative method only, not {method!r}.")

        regridder_class = factory.REGRIDDER_CLASSES[method]
        init_kwargs = {
            'source_grid': source_grid,
            'target_grid': target_grid,
            'device': factory.device,
            'parameter_manager': factory.parameter_manager,
            **kwargs
        }
        # Only add enforce_conservation if the regridder supports it
        if method == 'conservative' and 'enforce_conservation' in regridder_class.__init__.__code__.co_varnames:
            init_kwargs['enforce_conservation'] = factory.enforce_conservation

        # The cache key is the content fingerprint of everything that builds the regridder.
        cache_key = factory._create_cache_key(regridder_class, method, init_kwargs)
        cached = factory._regridder_cache.get(cache_key)
        if cached is not None:
            logger.debug("Using cached %s regridder %s", method, cache_key)
            return cached

        regridder = regridder_class(**init_kwargs)
        setattr(regridder, 'selected_method', method)
        factory._regridder_cache[cache_key] = regridder

        logger.info(
            f"Created {method} regridder: "
            f"{source_grid.shape} -> {target_grid.shape}"
        )

        return regridder

    def _select_method(self, metadata: Dict[str, Any]) -> str:
        """
        Read an explicit regridding method from variable metadata.
        
        Parameters:
            metadata : Dict[str, Any]
                Variable metadata
                
        Returns:
            str: Selected method name
        """
        if not isinstance(metadata, dict):
            raise TypeError("variable_metadata must be a mapping")
        method = metadata.get("regridding_method")
        if not isinstance(method, str) or method not in self.REGRIDDER_CLASSES:
            raise ValueError(
                "An explicit regridding_method from "
                f"{sorted(self.REGRIDDER_CLASSES)} is required"
            )
        return method
    
    def _create_cache_key(
        self,
        regridder_class: type,
        method: str,
        init_kwargs: Dict[str, Any],
    ) -> str:
        """
        Content fingerprint of everything that builds one regridder.

        The key covers the regridder class and method, every field of both GridDescriptors
        (grid type, coordinate system, dims, coordinates, connectivity, vertices, bounds, cell
        areas, mask, units, crs) and every other constructor argument by value (basis,
        tolerance, enforcement, device, sparsity, boundary settings and any other keyword), with
        the ParameterManager by identity: a same-shape grid of another geometry is another key.
        An argument without a canonical content encoding raises TypeError instead of being
        ignored. Parameter values read at construction come from the ParameterManager of the key
        (the cached regridder holds it, so its identity is not reused while the entry lives);
        after changing parameters of a live manager call clear_cache().

        Returns:
            str: SHA-256 hex digest (grid_utils.content_fingerprint)
        """
        manager = init_kwargs.get('parameter_manager')
        arguments = {name: value for name, value in init_kwargs.items() if name != 'parameter_manager'}
        return content_fingerprint(
            f"{regridder_class.__module__}.{regridder_class.__qualname__}",
            method,
            arguments,
            'none' if manager is None else f"parameter_manager:{id(manager)}",
        )

    def clear_cache(self):
        """Clear regridder cache to free memory."""
        self._regridder_cache.clear()
        logger.info("Cleared regridder cache")


class AutoRegridder(ScaleCouplingFunctor):
    """
    Automatic spatial regridder that selects method based on variable type and conservation need.

    Parameters
    ----------
    source_grid : GridDescriptor
        Source grid description (geometry/area in m^2).
    target_grid : GridDescriptor
        Target grid description (geometry/area in m^2).
    enforce_conservation : bool, optional
        Force conservative remapping for all variables (dimensionless flag).
    conservation_tolerance : float, optional
        Explicit relative tolerance of the conservative regridders [dimensionless]; None keeps
        the one regridder default coupling.scales.defaults.conservation_tolerance.
    device : torch.device, optional
        Target computation device.
    parameter_manager : ParameterManager, optional
        Parameter manager instance for tolerance/units lookup.
    strict_direction_inference : bool
        When True, require shape/metadata to unambiguously determine direction.
    validation_mode : Literal["strict"]
        Validation mode (strict only).
    """
    def __init__(
        self,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        enforce_conservation: bool = False,
        conservation_tolerance: Optional[float] = None,
        device: Optional[torch.device] = None,
        parameter_manager: Optional[Any] = None,
        strict_direction_inference: bool = True,
        validation_mode: Literal["strict"] = "strict",
    ):
        """
        Initialize automatic regridder.

        Parameters:
            source_grid : GridDescriptor
                Source grid description
            target_grid : GridDescriptor
                Target grid description
            enforce_conservation : bool
                If True, always use conservative remapping
            conservation_tolerance : float
                Tolerance for conservation checks
            device : Optional[torch.device]
                Computation device
            strict_direction_inference : bool
                When True, require shape/metadata to unambiguously determine direction.
            validation_mode : Literal["strict"]
                Validation mode (strict only).
        """
        metadata = FunctorMetadata(
            name="auto_regridder",
            version="1.0.0",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            description="Automatic spatial regridder selecting conservative/bilinear/nearest methods by metadata.",
            references=[
                "Jones, P.W. (1999). First- and second-order conservative remapping schemes for grids in spherical coordinates",
                "Ramshaw, J.D. (1985). Conservative rezoning algorithm for generalized two-dimensional meshes",
            ],
            mathematical_expression="y_j = (sum_i x_i * A_ij) / A_j",
            parameters={
                "enforce_conservation": "Enable mass/energy preserving remapping [dimensionless]",
                "conservation_tolerance": "Tolerance for conservation diagnostics [dimensionless]",
                "strict_direction_inference": "Require unambiguous direction inference [dimensionless]",
            },
            input_variables=["source_grid_fields"],
            output_variables=["target_grid_fields"],
            conservation_laws=["mass", "energy"] if enforce_conservation else [],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )

        super().__init__(
            name=metadata.name,
            metadata=metadata,
            config={},
            source_scale=f"grid_{source_grid.shape}",
            target_scale=f"grid_{target_grid.shape}",
            coupling_type="spatial",
            conservation_properties=metadata.conservation_laws,
            device=device,
            parameter_manager=parameter_manager,
        )

        self.source_grid = source_grid
        self.target_grid = target_grid
        self.enforce_conservation = enforce_conservation
        self.strict_direction_inference = bool(strict_direction_inference)

        if validation_mode != "strict":
            raise ValueError("validation_mode must be 'strict'.")
        self.validation_mode: Literal["strict"] = "strict"

        # Create factory (content-keyed regridder cache; an explicit tolerance or None for the
        # one regridder default)
        self.factory = SpatialRegridderFactory(
            enforce_conservation=enforce_conservation,
            conservation_tolerance=conservation_tolerance,
            device=device,
            parameter_manager=parameter_manager,
        )
        self.conservation_tolerance = self.factory.conservation_tolerance

        # Storage for metadata
        self._upscale_metadata: Optional[Dict[str, Any]] = None
        self._downscale_metadata: Optional[Dict[str, Any]] = None

        # Directional regridders (created on demand)
        self._source_to_target_regridder: Optional[ScaleCoupling] = None
        self._target_to_source_regridder: Optional[ScaleCoupling] = None
        self._current_method: Optional[str] = None
    
    
    def forward(self, state: Dict, time: float, dt: float, **kwargs) -> Dict:
        """
        Forward computation following Functor interface.

        Args:
            state: Current state variables
            time: Current simulation time [s]
            dt: Time step [s]
            **kwargs: Additional parameters
            
        Returns:
            Dict of computed fluxes/updates
        """
        # References: Jones (1999) and Ramshaw (1985) as cited in the module docstring; Shepard
        #   (1968) as listed in _regrid_metadata; RELIC ScaleCoupling contract
        #   (relic/core/functor/scale_coupling_bridge.py).
        # Mathematical expression: Y = K_m(X), K_m the regridder kernel chosen by the explicit
        #   metadata key regridding_method m in REGRIDDER_CLASSES; operation "identity" returns X.
        #   Direction is inferred from each tensor's trailing shape: equal to the source grid ->
        #   "downscale" if source cells > target cells else "upscale"; equal to the target grid
        #   swaps the grids; equal cell counts -> "identity".
        # Parameter meanings: X input tensor(s) from "data"; m method token; outputs keyed
        #   "regridded" (Tensor input) or by variable name, plus "_regrid_metadata".
        # Units: passed through unchanged; conversion and conservation belong to the kernel.
        # Domain/constraints: data None, no variable_metadata without cached regridders, an
        #   unknown regridding_method, non-Tensor values, no tensor entries, mixed inferred
        #   directions or origins, or an operation outside {upscale, downscale, identity} raise.
        # Sign convention: the sign of the source field is preserved by every selectable kernel.
        data = kwargs.get("data", state.get("data"))
        variables = kwargs.get("variables", state.get("variables"))
        variable_metadata = kwargs.get("variable_metadata", state.get("variable_metadata"))
        requested_operation = kwargs.get("operation", state.get("operation"))

        strict_inference = bool(self.strict_direction_inference)

        if data is None:
            raise ValueError("AutoRegridder.forward requires 'data' in kwargs or state.")

        # Choose method based on provided metadata; default stays cached if unchanged.
        if variable_metadata is not None:
            self.set_variable_metadata(variable_metadata)
        elif self._source_to_target_regridder is None or self._target_to_source_regridder is None:
            raise ValueError("AutoRegridder requires variable_metadata to select a regridding method.")

        source_shape = tuple(getattr(self.source_grid, "dims", getattr(self.source_grid, "shape", ())) or ())
        target_shape = tuple(getattr(self.target_grid, "dims", getattr(self.target_grid, "shape", ())) or ())

        def _num_cells(shape: Tuple[int, ...]) -> Optional[int]:
            if not shape:
                return None
            cells = 1
            for dim in shape:
                cells *= dim
            return cells

        src_cells = _num_cells(source_shape)
        tgt_cells = _num_cells(target_shape)

        def _strict_infer(shape: Tuple[int, ...]) -> Tuple[Optional[str], Optional[bool], str]:
            if not shape or not source_shape or not target_shape:
                return None, None, "insufficient_shape_metadata"

            matches_source = shape == source_shape
            matches_target = shape == target_shape

            if matches_source and matches_target:
                if src_cells is not None and tgt_cells is not None and src_cells != tgt_cells:
                    op = "downscale" if src_cells > tgt_cells else "upscale"
                else:
                    op = "identity"
                return op, False, "shape_matches_both_grids"

            if matches_source:
                if src_cells is None or tgt_cells is None:
                    return None, False, "missing_grid_cell_counts"
                if src_cells == tgt_cells:
                    return "identity", False, "equal_cell_counts"
                return ("downscale" if src_cells > tgt_cells else "upscale", False, "shape_matches_source_grid")

            if matches_target:
                if src_cells is None or tgt_cells is None:
                    return None, True, "missing_grid_cell_counts"
                if tgt_cells == src_cells:
                    return "identity", True, "equal_cell_counts"
                return ("downscale" if tgt_cells > src_cells else "upscale", True, "shape_matches_target_grid")

            return None, None, "shape_not_aligned_with_grids"

        def _infer(shape: Tuple[int, ...]) -> Tuple[Optional[str], Optional[bool], str]:
            return _strict_infer(shape)

        shape_records: List[Dict[str, Any]] = []

        def _extract_shape(tensor: Tensor) -> Tuple[int, ...]:
            if source_shape and tensor.dim() >= len(source_shape):
                return tuple(tensor.shape[-len(source_shape):])
            return tuple(tensor.shape)

        def _record(var_name: str, tensor: Tensor) -> None:
            tensor_shape = _extract_shape(tensor)
            op, flag, reason = _infer(tensor_shape)
            shape_records.append(
                {
                    "var": var_name,
                    "shape": tensor_shape,
                    "operation": op,
                    "input_is_target": flag,
                    "reason": reason,
                }
            )

        if isinstance(data, torch.Tensor):
            _record("data", data)
        elif isinstance(data, dict):
            target_vars = variables or list(data.keys())
            for var in target_vars:
                if var not in data:
                    continue
                value = data[var]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"AutoRegridder expects Tensor values for variable '{var}'")
                _record(var, value)
            if not shape_records:
                raise ValueError("AutoRegridder received no tensor entries to regrid.")
        else:
            raise TypeError(f"Unsupported data type for AutoRegridder: {type(data).__name__}")

        allowed_ops = {"upscale", "downscale", "identity"}
        if requested_operation is not None and requested_operation not in allowed_ops:
            raise ValueError(
                f"Invalid operation '{requested_operation}'. Must be one of {sorted(allowed_ops)} or None for inference."
            )

        inferred_ops = [rec["operation"] for rec in shape_records if rec["operation"] is not None]
        inferred_flags = [rec["input_is_target"] for rec in shape_records if rec["input_is_target"] is not None]

        operation = requested_operation
        input_is_target = False

        if operation is None:
            if not inferred_ops:
                raise ValueError("AutoRegridder cannot infer operation from shapes; pass 'operation' explicitly or align tensors to source/target grids.")
            if len(set(inferred_ops)) > 1:
                raise ValueError("AutoRegridder detected mixed inferred directions across variables; provide explicit 'operation' or split the call per direction.")
            operation = inferred_ops[0]

            if inferred_flags:
                if len(set(inferred_flags)) > 1:
                    raise ValueError("AutoRegridder detected mixed grid origins for inputs; align shapes with a single grid or call with explicit operation.")
                input_is_target = inferred_flags[0]
            else:
                raise ValueError("AutoRegridder cannot determine grid origin for provided data; supply grid-aligned tensors.")
        else:
            if inferred_ops:
                mismatched = [op for op in inferred_ops if op not in ("identity", operation)]
                if mismatched:
                    raise ValueError(
                        f"Requested operation '{operation}' conflicts with inferred {set(mismatched)} based on tensor shapes."
                    )

            if inferred_flags:
                if len(set(inferred_flags)) > 1:
                    raise ValueError(
                        "AutoRegridder found mixed grid origins while an explicit operation was provided; split calls or align inputs."
                    )
                input_is_target = inferred_flags[0]
            else:
                if operation != "identity":
                    raise ValueError(
                        "AutoRegridder cannot determine whether data is on source_grid or target_grid; provide grid-aligned tensors."
                    )
                input_is_target = False

        def _select_regridder(op: str, input_from_target: bool) -> Optional[ScaleCoupling]:
            if op == "identity":
                return None

            # The regridder built from the input's grid to the other grid.
            cache_attr = "_target_to_source_regridder" if input_from_target else "_source_to_target_regridder"
            cached = getattr(self, cache_attr)
            if cached is None:
                # forward sets the method from the variable metadata before this point
                # (set_variable_metadata); no method is assumed.
                raise ValueError(
                    "AutoRegridder has no regridder for this direction: declare variable metadata "
                    "with its regridding_method (set_variable_metadata) first."
                )
            return cached

        outputs: Dict[str, Tensor] = {}

        def _apply(tensor: Tensor) -> Tensor:
            if operation == "identity":
                return tensor
            selected_regridder = _select_regridder(operation, input_is_target)
            # The selected regridder is built from the input's grid to the other grid, so its
            # forward mapping (upscale: its source grid -> its target grid, for every class) is the
            # mapping of this call whatever the direction label; the nearest-neighbour downscale
            # would map the other way.
            return selected_regridder.upscale(tensor)

        if isinstance(data, torch.Tensor):
            outputs["regridded"] = _apply(data)
        else:
            target_vars = variables or list(data.keys())
            for var in target_vars:
                if var not in data:
                    continue
                outputs[var] = _apply(data[var])

        outputs["_regrid_metadata"] = {
            "operation": operation,
            "method": "identity" if operation == "identity" else self._current_method,
            "source_scale": getattr(self.source_grid, "shape", None),
            "target_scale": getattr(self.target_grid, "shape", None),
            "direction_mode": "strict" if strict_inference else "lenient",
            "validation_mode": self.validation_mode,
            "references": [
                "Jones (1999) conservative remapping",
                "Shepard (1968) nearest neighbor interpolation",
            ],
        }
        return outputs

    def set_variable_metadata(self, metadata: Dict[str, Any]):
        """
        Set variable metadata for automatic method selection.
        
        Parameters:
            metadata : Dict[str, Any]
                Variable metadata including type, units, etc.
        """
        self._upscale_metadata = metadata
        self._downscale_metadata = metadata
        
        # Select and create regridder
        method = self.factory._select_method(metadata)
        if (
            method != self._current_method
            or self._source_to_target_regridder is None
            or self._target_to_source_regridder is None
        ):
            self._source_to_target_regridder = self.factory.create_regridder(
                self.source_grid,
                self.target_grid,
                method=method,
                variable_metadata=metadata,
            )
            self._target_to_source_regridder = self.factory.create_regridder(
                self.target_grid,
                self.source_grid,
                method=method,
                variable_metadata=metadata,
            )
            self._current_method = method
    
    def upscale(self, fine_data: Tensor) -> Tensor:
        """
        Map a field on source_grid to target_grid with the declared method.

        Parameters:
            fine_data : Tensor
                Field on source_grid

        Returns:
            Tensor: Field on target_grid

        Raises:
            ValueError: when no method is declared (set_variable_metadata); none is assumed.
        """
        if self._source_to_target_regridder is None:
            raise ValueError(
                "AutoRegridder.upscale needs variable metadata with its regridding_method "
                "(set_variable_metadata); the method is never assumed."
            )
        return self._source_to_target_regridder.upscale(fine_data)

    def downscale(self, coarse_data: Tensor) -> Tensor:
        """
        Map a field on target_grid to source_grid with the declared method.

        The target-to-source regridder is built from target_grid to source_grid, so its forward
        mapping (``upscale``, the source-to-target mapping of every regridder class) is used.

        Parameters:
            coarse_data : Tensor
                Field on target_grid

        Returns:
            Tensor: Field on source_grid

        Raises:
            ValueError: when no method is declared (set_variable_metadata); none is assumed.
        """
        if self._target_to_source_regridder is None:
            raise ValueError(
                "AutoRegridder.downscale needs variable metadata with its regridding_method "
                "(set_variable_metadata); the method is never assumed."
            )
        return self._target_to_source_regridder.upscale(coarse_data)

    def get_current_method(self) -> Optional[str]:
        """Get currently selected regridding method."""
        return self._current_method
    
    def clear_cache(self):
        """Clear internal caches."""
        self.factory.clear_cache()
        self._source_to_target_regridder = None
        self._target_to_source_regridder = None
        self._current_method = None



