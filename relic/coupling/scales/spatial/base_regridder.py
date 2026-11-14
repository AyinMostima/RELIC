"""
Shared spatial regridding workflow.

Provides a unified forward pipeline for spatial regridders to avoid duplicated
operation detection, input validation, and output formatting logic.

Mathematical Expression:
    Generic remapping:
        Y[J] = R(X[i], W[i, J], method)
    where R is the method-specific remapping implemented in subclasses.

References:
    - Jones, P.W. (1999). Conservative remapping schemes for climate models.
    - Shepard, D. (1968). Two-dimensional interpolation function.
    - ESMF Regridding Guide (spatial remapping workflow).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from relic.core.abstractions import FluxDict, StateDict
from relic.core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from relic.core.functor import ScaleCouplingFunctor
from relic.coupling.core.conservation_checker import require_relative_tolerance
from relic.coupling.scales.spatial.grid_utils import GridDescriptor
from relic.parameters.namespaces import ParameterNamespace, resolve_namespace

logger = logging.getLogger(__name__)


class SpatialRegridderBase(ScaleCouplingFunctor, ABC):
    """
    Base class for spatial regridders with shared forward() logic.

    Responsibilities:
    - Detect operation direction (upscale/downscale) when not provided.
    - Validate and cast input data to tensors on the correct device.
    - Format outputs consistently while allowing subclasses to extend metadata.

    Subclasses must implement the actual remapping kernels via ``upscale`` and
    ``downscale``.
    """

    def __init__(
        self,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        *,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[FunctorMetadata] = None,
        parameter_manager: Optional[Any] = None,
        enforce_conservation: Optional[bool] = None,
        conservation_tolerance: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        meta = metadata or FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Base spatial regridder",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression="Y[J] = R(X[i], W[i,J], method)",
            references=[
                "Jones, P.W. (1999). Conservative remapping schemes for climate models.",
                "ESMF Regridding Guide.",
            ],
            parameters={
                "enforce_conservation": "Enable conservative remapping checks [dimensionless]",
                "conservation_tolerance": "Tolerance for conservation diagnostics [dimensionless]",
                "operation": "Requested remapping direction (upscale/downscale) [dimensionless]",
            },
            input_variables=["data"],
            output_variables=["data"],
            conservation_laws=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )

        def _grid_scale_label(grid: Any) -> str:
            coord = getattr(grid, "coord_system", None) or "cartesian"
            dims = getattr(grid, "dims", None)
            if dims is None:
                shape = getattr(grid, "shape", None)
                if shape is not None:
                    dims = tuple(shape)
                else:
                    dims = "unknown"
            return f"{coord}_{dims}"

        super().__init__(
            name=name,
            config=config or {},
            metadata=meta,
            parameter_manager=parameter_manager,
            source_scale=_grid_scale_label(source_grid),
            target_scale=_grid_scale_label(target_grid),
            coupling_type="spatial",
            **kwargs,
        )

        self.source_grid = source_grid
        self.target_grid = target_grid

        # Parameter-managed defaults with explicit validation
        if enforce_conservation is not None:
            self.enforce_conservation = bool(enforce_conservation)
        else:
            pm_enforce = self.get_parameter_value(
                "enforce_conservation",
                namespace=resolve_namespace(ParameterNamespace.COUPLING_SCALES_SPATIAL),
                units="dimensionless",
                validate=True,
                required=False,
            )
            if pm_enforce is not None:
                self.enforce_conservation = bool(
                    pm_enforce.detach().cpu().item() if isinstance(pm_enforce, Tensor) else pm_enforce
                )
            else:
                raise KeyError(
                    "Required parameter coupling.scales.spatial.enforce_conservation is missing; "
                    "provide it via ParameterManager or pass enforce_conservation explicitly."
                )

        # One tolerance source per regridder: the explicit argument (the ScaleManager facade passes
        # coupling.scales.defaults.precision.<mode>.tolerance of the call's mode), else the one
        # default coupling.scales.defaults.conservation_tolerance, read here for every regridder
        # class and every entry (direct construction or a factory); both must be finite in (0, 1).
        if conservation_tolerance is not None:
            self.conservation_tolerance = require_relative_tolerance(
                conservation_tolerance, "conservation_tolerance"
            )
        else:
            pm_tol = self.get_parameter_value(
                "conservation_tolerance",
                namespace=resolve_namespace(ParameterNamespace.COUPLING_SCALES_DEFAULTS),
                units="dimensionless",
                validate=True,
                required=False,
            )
            if pm_tol is None:
                raise KeyError(
                    "Required parameter coupling.scales.defaults.conservation_tolerance is missing; "
                    "provide it via ParameterManager or pass conservation_tolerance explicitly."
                )
            self.conservation_tolerance = require_relative_tolerance(
                float(pm_tol.detach().cpu().item()) if isinstance(pm_tol, Tensor) else pm_tol,
                "coupling.scales.defaults.conservation_tolerance",
            )

        self.metadata.conservation_laws = ["mass"] if self.enforce_conservation else []
        self.conservation_properties = list(self.metadata.conservation_laws)

        self._move_grids_to_device()

    def _move_grids_to_device(self) -> None:
        """Move grid tensors to the functor device."""
        for grid in (self.source_grid, self.target_grid):
            grid.x_coords = grid.x_coords.to(self.device)
            grid.y_coords = grid.y_coords.to(self.device)
            if grid.x_bounds is not None:
                grid.x_bounds = grid.x_bounds.to(self.device)
            if grid.y_bounds is not None:
                grid.y_bounds = grid.y_bounds.to(self.device)
            if grid.cell_areas is not None:
                grid.cell_areas = grid.cell_areas.to(self.device)
            if grid.mask is not None:
                grid.mask = grid.mask.to(self.device)

    def _resolve_grids(
        self,
        state: StateDict,
        **_: Any,
    ) -> tuple[GridDescriptor, GridDescriptor]:
        """Allow subclasses to override grid resolution logic."""
        if "source_grid" in state:
            source_grid = state["source_grid"]
        else:
            source_grid = self.source_grid
        if "target_grid" in state:
            target_grid = state["target_grid"]
        else:
            target_grid = self.target_grid
        return source_grid, target_grid

    def _detect_operation(
        self,
        state: StateDict,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        **_: Any,
    ) -> str:
        """
        Determine whether to upscale (coarse -> fine) or downscale (fine -> coarse).

        If the caller provides an explicit operation it is validated; otherwise
        the decision is based on relative grid sizes.
        """
        if "operation" in state:
            operation = state["operation"]
        else:
            operation = None
        if operation is not None:
            if operation not in ("upscale", "downscale"):
                raise ValueError(f"Invalid operation: {operation}")
            return operation

        source_size = self._grid_cell_count(source_grid)
        target_size = self._grid_cell_count(target_grid)

        if target_size < source_size:
            return "downscale"
        if target_size > source_size:
            return "upscale"
        # Equal cell counts decide no direction; the call names it.
        raise ValueError(
            f"Source and target grids have the same number of cells ({source_size}); pass "
            "operation='upscale' or 'downscale' explicitly (no direction is assumed)."
        )

    def _extract_data(
        self,
        state: StateDict,
        source_grid: GridDescriptor,
        **kwargs: Any,
    ) -> Tensor:
        """Extract and validate input data from state or kwargs."""
        if "data" in state:
            data = state["data"]
        elif "data" in kwargs:
            data = kwargs["data"]
        else:
            data = None
        if data is None:
            raise ValueError("Spatial regridding requires 'data' in state or kwargs.")

        if isinstance(data, Tensor):
            data_tensor = data.to(self.device)
        else:
            data_tensor = torch.as_tensor(data, device=self.device, dtype=torch.float32)
        spatial_dims = tuple(source_grid.dims)
        trailing_dims: tuple[int, ...] = ()
        if data_tensor.dim() >= len(spatial_dims):
            trailing_dims = tuple(data_tensor.shape[-len(spatial_dims) :])

        if trailing_dims != spatial_dims:
            expected_size = self._grid_cell_count(source_grid)
            actual_size = data_tensor.numel()
            if actual_size % expected_size == 0 and expected_size > 0:
                batch = actual_size // expected_size
                new_shape = (batch, *spatial_dims) if batch > 1 else spatial_dims
                data_tensor = data_tensor.reshape(new_shape)
            else:
                raise ValueError(
                    f"Input data of shape {tuple(data_tensor.shape)} has {actual_size} values; the source "
                    f"grid has {expected_size} cells {spatial_dims}, so the field is neither one field "
                    "nor a batch of fields on that grid."
                )
        return data_tensor

    def _grid_cell_count(self, grid: GridDescriptor) -> int:
        """Compute total cell count for the provided grid."""
        total = 1
        for dim in grid.dims:
            total *= dim
        return total

    def _prepare_run(
        self,
        state: StateDict,
        operation: str,
        data: Tensor,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        **kwargs: Any,
    ) -> None:
        """Hook for subclasses to adjust internal flags before execution."""
        _ = (state, operation, data, source_grid, target_grid, kwargs)

    def _execute_operation(
        self,
        data: Tensor,
        operation: str,
        **kwargs: Any,
    ) -> Tensor:
        """Dispatch to subclass-specific kernels."""
        if operation == "downscale":
            return self.downscale(data, **kwargs)
        return self.upscale(data, **kwargs)

    def _format_output(
        self,
        result: Tensor,
        operation: str,
        state: StateDict,
        *,
        input_data: Tensor,
        source_grid: GridDescriptor,
        target_grid: GridDescriptor,
        **_: Any,
    ) -> FluxDict:
        """Default output formatter; subclasses can override for richer diagnostics."""
        return {
            "data": result,
            "operation": operation,
            "source_grid": source_grid,
            "target_grid": target_grid,
            "method": self.__class__.__name__,
            "metadata": state["metadata"] if "metadata" in state else {},
            "input_shape": tuple(input_data.shape),
        }

    def forward(self, state: StateDict, time: float, dt: float, **kwargs: Any) -> FluxDict:
        """
        Unified forward workflow for spatial regridders.

        Steps:
            1. Resolve grids and detect operation.
            2. Extract and validate input data.
            3. Execute subclass-specific regridding.
            4. Format outputs and diagnostics.

        Returns
        -------
        FluxDict
            Structured output including regridded data and metadata.
        """
        # References: Jones (1999), Shepard (1968), and the ESMF Regridding Guide as cited in the
        #   module docstring; RELIC ScaleCoupling contract (relic/core/functor/scale_coupling_bridge.py).
        # Mathematical expression: Y[J] = R(X[i], W[i,J], method) where R is the subclass kernel
        #   reached through _execute_operation -> self.downscale (operation "downscale") or
        #   self.upscale (operation "upscale"); this base applies no arithmetic of its own.
        # Parameter meanings: X[i] source-grid values from state["data"] or kwargs["data"];
        #   W[i,J] subclass weights; Y[J] target-grid values returned under "data" together with
        #   "operation", "source_grid", "target_grid", "method", "metadata", and "input_shape".
        # Units: passed through unchanged; any conversion or conservation belongs to the kernel.
        # Domain/constraints: state["operation"] must be "upscale" or "downscale" (else ValueError);
        #   otherwise fewer target cells -> "downscale", more -> "upscale", equal cell counts raise
        #   (the call names the operation); data numel must be a multiple of the source cell count (else
        #   ValueError); missing data and every kernel error raise to the caller (no result is
        #   returned for a failed call).
        # Sign convention: not applicable here; the sign of the source field is passed to the kernel.
        source_grid, target_grid = self._resolve_grids(state, **kwargs)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in {"data", "operation"}}
        data = self._extract_data(state, source_grid, **kwargs)
        operation = self._detect_operation(state, source_grid, target_grid, **kwargs)
        self._prepare_run(state, operation, data, source_grid, target_grid, **clean_kwargs)
        result = self._execute_operation(data, operation, **clean_kwargs)

        return self._format_output(
            result,
            operation,
            state,
            input_data=data,
            source_grid=source_grid,
            target_grid=target_grid,
            **clean_kwargs,
        )

    @abstractmethod
    def upscale(self, data: Tensor, **kwargs: Any) -> Tensor:
        """Coarse-to-fine remapping implemented by subclasses."""

    @abstractmethod
    def downscale(self, data: Tensor, **kwargs: Any) -> Tensor:
        """Fine-to-coarse remapping implemented by subclasses."""
