"""
Biological-process surrogate models for scientific process emulation.

This module adapts machine-learning emulators to the existing
``BiologicalProcessFunctor`` lifecycle. It does not provide a category-neutral
surrogate bridge: physical-field, transport, and scale-coupling emulators must
use their respective canonical bridges.

Key Features:
    - Biological-process surrogate interface via SurrogateFunctor
    - ONNX Runtime integration for cross-platform deployment
    - Automatic input/output variable mapping with CF standard names
    - Explicit conservation-ledger validation
    - Uncertainty quantification and error propagation
    - Batch processing for spatial/ensemble simulations

Architecture:
    SurrogateFunctor (Abstract Base)
        ONNXSurrogate (ONNX Runtime implementation)
        GaussianProcessSurrogate (future implementation)  
        PhysicsInformedSurrogate (future implementation)

Mathematical Foundation:
    A surrogate approximates a declared biological process as
    ``Y = S(X; theta)``. When conservation checking is enabled, the configured
    ledger matrix ``C`` is evaluated as ``R = Y C^T`` and the call is accepted
    only when ``max(abs(R)) <= epsilon``. Output names never define physics.

    Inputs are scalar or batch vectors. Spatial aggregation is owned by the
    existing scale-coupling layer and is never inferred here.
    Uncertainty: sigma^2_total = sigma^2_approx + sigma^2_param + sigma^2_input

References:
    - Sacks, J. et al. (1989). Design and Analysis of Computer Experiments
    - Rasp, S. et al. (2018). Deep learning to represent subgrid processes
    - Raissi, M. et al. (2019). Physics-informed neural networks
    - Kennedy, M.C. & O'Hagan, A. (2001). Bayesian calibration of computer models

Usage:
    >>> from relic.models.surrogates import ONNXSurrogate
    >>> from relic.core.functor import FunctorMetadata
    >>> 
    >>> # Configure surrogate for photosynthesis
    >>> config = {
    >>>     'model_path': 'models/photosynthesis_farquhar.onnx',
    >>>     'input_variables': ['air_temperature', 'photosynthetic_photon_flux', 'atmospheric_co2'],
    >>>     'output_variables': ['net_photosynthesis', 'stomatal_conductance'],
    >>>     'conservation_check': False,  # Photosynthesis is not conservative
    >>>     'device': 'cpu'
    >>> }
    >>> 
    >>> metadata = FunctorMetadata(
    >>>     name="photosynthesis_surrogate",
    >>>     description="ONNX surrogate for Farquhar photosynthesis model"
    >>> )
    >>> 
    >>> surrogate = ONNXSurrogate("photosynthesis_surrogate", config, metadata)
    >>> updates = surrogate(state, time=3600.0, dt=60.0)
"""

import importlib
import importlib.util
import logging
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

import torch
from relic.core import device_policy
from torch import Tensor

# Import core RELIC components
from ..core.functor import (
    BiologicalProcessFunctor, FunctorMetadata, ProcessCategory,
    ExecutionPhase, ConfigDict,
)
from ..core.state import UpdateItem, UpdateMode
from ..core.abstractions import StateDict, FluxDict, Device
from ..core.variable_mapper import variable_name_mapper as variable_mapper
from ..data.standards.unit_converter import unit_converter, ConversionContext
from .config_utils import coerce_config_dict

# ONNX Runtime is a required dependency for ONNX surrogate support.
import onnxruntime as ort
ONNX_AVAILABLE = True

logger = logging.getLogger(__name__)

# Type aliases for surrogate models
ModelPath = Union[str, Path]
VariableMapping = Dict[str, str]
ConservationFunction = Callable[[FluxDict], Union[Tensor, float]]
UncertaintyEstimator = Callable[[Tensor], Tensor]

class SurrogateError(Exception):
    """Base exception for surrogate model errors."""
    pass


class ModelLoadError(SurrogateError):
    """Raised when surrogate model loading fails."""
    pass


class InferenceError(SurrogateError):
    """Raised when surrogate model inference fails."""
    pass


class ConservationViolationError(SurrogateError):
    """Raised when conservation laws are violated beyond tolerance."""
    pass


class SurrogateFunctor(BiologicalProcessFunctor):
    """
    Abstract base class for biological-process surrogate functors.
    
    This class is intentionally bound to ``BiologicalProcessFunctor``. A model
    with physical-field, transport, or scale-coupling semantics must use the
    corresponding existing bridge instead of relabeling this class.
    
    This class handles common surrogate model functionality including:
    - Input/output variable mapping and unit conversion
    - Conservation law checking and enforcement
    - Uncertainty quantification interfaces
    - Batch processing capabilities
    - Error handling and validation
    
    Subclasses must implement:
    - _load_model(): Load the surrogate model from storage
    - _predict(): Run inference on prepared inputs
    - _estimate_uncertainty(): Estimate prediction uncertainty (optional)
    
    Mathematical Expression:
        Forward pass: F_surrogate(s, t, dt) -> DeltaS
        where s in StateSpace, t in Time, dt in R+, DeltaS in FluxSpace
        
        Conservation ledger: R = Y C^T, max(abs(R)) <= epsilon
        Uncertainty: sigma(DeltaS) = f(model_uncertainty, input_uncertainty)
        
    Parameters:
        name : str
            Unique functor identifier
        config : ConfigDict
            Configuration including model path, variables, options
        metadata : FunctorMetadata
            Scientific documentation and process metadata
        
    Required Config Keys:
        model_path : str
            Path to surrogate model file
        input_variables : List[str]
            Required state variables for inference
        output_variables : List[str]
            Output flux/rate variables
            
    Optional Config Keys:
        input_mapping : Dict[str, str]
            Map state variables to model input names
        output_mapping : Dict[str, str]
            Map model outputs to flux variable names
        units_conversion : Dict[str, Tuple[str, str]]
            Unit conversions: variable -> (from_unit, to_unit)
        conservation_check : bool
            Whether to enforce a declared conservation ledger (default: False)
        conservation_matrix : array-like, optional
            Matrix C with shape [n_constraints, n_outputs]. Required at runtime
            when conservation checking is enabled and no callback is registered.
        conservation_tolerance : float
            Maximum allowed conservation error (default: 1e-6)
        device : str
            Computation device: 'cpu', 'cuda', 'auto' (default: 'auto')
        batch_size : int
            Maximum batch size for inference (default: 1)
        uncertainty_quantification : bool
            Whether to estimate prediction uncertainty (default: False)
        scale_rate_outputs_by_dt : bool
            Scale only explicitly declared rate outputs by dt (default: False)
        rate_output_variables : List[str]
            Output variables that have rate semantics and may be scaled by dt
    
    References:
        - O'Hagan, A. (2006). Bayesian analysis of computer code outputs
        - Razavi, S. et al. (2012). Review of surrogate modeling in water resources
    """
    
    def __init__(self, 
                 name: str,
                 config: ConfigDict,
                 metadata: Optional[FunctorMetadata] = None,
                 **kwargs) -> None:
        """Initialize surrogate functor with configuration."""

        settings = coerce_config_dict(config)
        SurrogateFunctor._validate_config(settings)
        configured_outputs = list(settings["output_variables"])
        configured_output_mapping = dict(settings.get("output_mapping", {}))
        runtime_outputs = [
            configured_output_mapping.get(variable, variable)
            for variable in configured_outputs
        ]

        # Set default metadata for surrogate models
        if metadata is None:
            metadata = FunctorMetadata(
                name=name,
                description=f"Surrogate model functor: {name}",
                category=ProcessCategory.BIOLOGICAL_PROCESS,
                execution_phase=ExecutionPhase.MAIN_PROCESSES,
                input_variables=list(settings.get("input_variables", [])),
                output_variables=runtime_outputs,
                references=[
                    "O'Hagan, A. (2006). Bayesian analysis of computer code outputs: "
                    "A tutorial. Reliability Engineering and System Safety 91:1290-1300.",
                    "Razavi, S., Tolson, B. A., and Burn, D. H. (2012). Review of "
                    "surrogate modeling in water resources. Water Resources Research "
                    "48:W07401.",
                ],
            )
        else:
            category = metadata.category
            category_value = (
                category.value if isinstance(category, ProcessCategory) else str(category)
            )
            if category_value != ProcessCategory.BIOLOGICAL_PROCESS.value:
                raise TypeError(
                    "SurrogateFunctor requires biological_process metadata; "
                    "use the canonical bridge for the declared process category"
                )
            metadata.category = ProcessCategory.BIOLOGICAL_PROCESS
            metadata.output_variables = runtime_outputs
            metadata.optional_output_variables = [
                configured_output_mapping.get(variable, variable)
                for variable in metadata.optional_output_variables
            ]
        
        super().__init__(name, config, metadata, **kwargs)
        
        # Model and inference components
        self.model_path: ModelPath = Path(settings['model_path'])
        self.model: Any = None  # Loaded model (type depends on implementation)
        self.model_loaded: bool = False
        
        # Variable mapping and conversion
        self.input_variables: List[str] = settings['input_variables']
        self.output_variables: List[str] = settings['output_variables']
        self.input_mapping: VariableMapping = settings.get('input_mapping', {})
        self.output_mapping: VariableMapping = settings.get('output_mapping', {})
        self.units_conversion: Dict[str, Tuple[str, str]] = settings.get('units_conversion', {})
        
        # Conservation and validation settings
        self.conservation_check: bool = settings.get('conservation_check', False)
        self.conservation_tolerance: Optional[float] = settings.get('conservation_tolerance')
        self.conservation_functions: List[ConservationFunction] = []
        self.scale_rate_outputs_by_dt: bool = settings.get(
            'scale_rate_outputs_by_dt', False
        )
        self.rate_output_variables: Tuple[str, ...] = tuple(
            settings.get('rate_output_variables', ())
        )
        output_update_mode = settings.get('output_update_mode', None)
        self.output_update_mode: Optional[UpdateMode] = (
            UpdateMode(output_update_mode) if output_update_mode is not None else None
        )
        
        # Device and performance settings
        self.device_preference: str = settings.get('device', 'auto')
        self.device: Device = self._resolve_device()
        self.batch_size: int = settings.get('batch_size', 1)
        conservation_matrix = settings.get('conservation_matrix')
        matrix_tensor = (
            None
            if conservation_matrix is None
            else torch.as_tensor(
                conservation_matrix,
                dtype=torch.float32,
                device=self.device,
            )
        )
        self.register_buffer('_conservation_matrix', matrix_tensor, persistent=True)
        
        # Uncertainty quantification
        self.uncertainty_quantification: bool = settings.get('uncertainty_quantification', False)
        self.uncertainty_estimator: Optional[UncertaintyEstimator] = None
        
        # Caching for performance
        self._input_tensor_cache: Optional[Tensor] = None
        self._output_tensor_cache: Optional[Tensor] = None
        
        # Load the surrogate model
        self._load_model()
        
        logger.info(f"Initialized surrogate functor: {name}")
    
    @staticmethod
    def _validate_config(settings: ConfigDict) -> None:
        """Validate surrogate configuration parameters."""
        required_keys = ['model_path', 'input_variables', 'output_variables']
        
        for key in required_keys:
            if key not in settings:
                raise ValueError(f"Missing required config key: {key}")
        
        # The model file must exist unless the configuration declares
        # validate_model_path = false (a checkpoint that is still to be
        # created). No path name ('mock', 'test') relaxes this check.
        validate_path = settings.get('validate_model_path', True)
        if not isinstance(validate_path, bool):
            raise TypeError("validate_model_path must be a boolean")
        if validate_path:
            model_path = Path(settings['model_path'])
            if not model_path.exists():
                raise FileNotFoundError(f"Surrogate model file not found: {model_path}")
        
        # Validate variable lists
        if not settings['input_variables']:
            raise ValueError("input_variables cannot be empty")
        if not settings['output_variables']:
            raise ValueError("output_variables cannot be empty")

        output_variables = list(settings["output_variables"])
        output_mapping = settings.get("output_mapping", {})
        if not isinstance(output_mapping, dict):
            raise TypeError("output_mapping must be a dictionary")
        for source_name, target_name in output_mapping.items():
            if source_name not in output_variables:
                raise ValueError(
                    f"Output mapping variable '{source_name}' not in output_variables"
                )
            if not isinstance(target_name, str) or not target_name.strip():
                raise ValueError(
                    "Output mapping targets must be non-empty variable names"
                )
        runtime_output_variables = [
            output_mapping.get(variable, variable)
            for variable in output_variables
        ]
        if len(runtime_output_variables) != len(set(runtime_output_variables)):
            raise ValueError("Output mapping targets must be unique")

        for option in ('conservation_check', 'scale_rate_outputs_by_dt'):
            value = settings.get(option, False)
            if not isinstance(value, bool):
                raise TypeError(f"{option} must be a boolean")

        rate_output_variables = settings.get('rate_output_variables', ())
        if not isinstance(rate_output_variables, (list, tuple)):
            raise TypeError("rate_output_variables must be a list or tuple")
        if any(
            not isinstance(variable, str) or not variable.strip()
            for variable in rate_output_variables
        ):
            raise ValueError("rate_output_variables must contain non-empty names")
        if len(rate_output_variables) != len(set(rate_output_variables)):
            raise ValueError("rate_output_variables must be unique")
        unknown_rate_outputs = set(rate_output_variables) - set(output_variables)
        if unknown_rate_outputs:
            raise ValueError(
                "rate_output_variables must be declared in output_variables: "
                f"{sorted(unknown_rate_outputs)}"
            )
        if settings.get('scale_rate_outputs_by_dt', False) and not rate_output_variables:
            raise ValueError(
                "scale_rate_outputs_by_dt requires explicit rate_output_variables"
            )

        # Validate a declared conservation tolerance (absolute residual of
        # the declared ledger, in the output units); none is invented.
        if 'conservation_tolerance' in settings:
            tolerance_tensor = torch.as_tensor(
                settings['conservation_tolerance'], dtype=torch.float64
            )
            if tolerance_tensor.numel() != 1 or not bool(
                torch.isfinite(tolerance_tensor).all().item()
            ):
                raise ValueError("conservation_tolerance must be a finite scalar")
            if bool((tolerance_tensor < 0).item()):
                raise ValueError("conservation_tolerance must be non-negative")

        conservation_matrix = settings.get('conservation_matrix')
        if conservation_matrix is not None:
            try:
                matrix = torch.as_tensor(conservation_matrix, dtype=torch.float64)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(
                    "conservation_matrix must be a finite numeric matrix"
                ) from exc
            if matrix.ndim != 2 or matrix.numel() == 0:
                raise ValueError(
                    "conservation_matrix must be a non-empty rank-2 matrix"
                )
            if matrix.shape[1] != len(output_variables):
                raise ValueError(
                    "conservation_matrix column count must match output_variables"
                )
            if not bool(torch.isfinite(matrix).all().item()):
                raise ValueError("conservation_matrix must contain finite values")
            if bool(torch.all(matrix == 0, dim=1).any().item()):
                raise ValueError("conservation_matrix rows must be non-zero")

        # A conservation check compares the ledger residual with a declared
        # tolerance; there is no default tolerance.
        if settings.get('conservation_check', False) and 'conservation_tolerance' not in settings:
            raise ValueError(
                "conservation_check requires an explicit conservation_tolerance "
                "(absolute ledger residual in the output units)"
            )
    
    def _resolve_device(self) -> Device:
        """Resolve computation device based on preference and availability."""
        preference = str(self.device_preference).strip().lower()
        explicit_device: Optional[str]
        if preference in {"", "auto", "default"}:
            explicit_device = None
        else:
            explicit_device = preference
        return device_policy.resolve_device(
            explicit_device=explicit_device,
            module_name=self.__class__.__module__,
            default="auto",
        )
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the surrogate model from storage.
        
        This method must be implemented by subclasses to load their specific
        model format (ONNX, PyTorch, scikit-learn, etc.).
        
        Should set self.model and self.model_loaded = True on success.
        Should raise ModelLoadError on failure.
        """
        pass
    
    @abstractmethod
    def _predict(self, input_tensor: Tensor) -> Tensor:
        """Run inference on prepared input tensor.
        
        Parameters:
            input_tensor : Tensor
                Prepared model inputs, shape [batch_size, n_features]
                
        Returns:
            Tensor: Model predictions, shape [batch_size, n_outputs]
            
        Raises:
            InferenceError: If inference fails
        """
        pass
    
    def _estimate_uncertainty(self, input_tensor: Tensor) -> Optional[Tensor]:
        """Estimate prediction uncertainty (optional).
        
        Default implementation returns None (no uncertainty).
        Subclasses can override to provide uncertainty estimates.
        
        Parameters:
            input_tensor : Tensor
                Model inputs, shape [batch_size, n_features]
                
        Returns:
            Optional[Tensor]: Uncertainty estimates, shape [batch_size, n_outputs]
                            None if uncertainty quantification not available
        """
        return None
    
    def _prepare_input(self, state: StateDict) -> Tensor:
        """Prepare model input tensor from state dictionary.
        
        Performs variable mapping, unit conversion, and tensor preparation.
        
        Parameters:
            state : StateDict
                Current system state
                
        Returns:
            Tensor: Prepared input tensor, shape [1, n_features]
            
        Raises:
            ValueError: If required variables are missing
            SurrogateError: If preparation fails
        """
        try:
            input_columns: List[Tensor] = []
            
            for var_name in self.input_variables:
                state_var_name = var_name
                if var_name not in state:
                    if var_name in self.input_mapping:
                        mapped_name = self.input_mapping[var_name]
                        if mapped_name in state:
                            state_var_name = mapped_name
                        else:
                            raise KeyError(f"Required input variable not found (mapped): {mapped_name}")
                    else:
                        # Try CF standard name mapping
                        mapped_name = variable_mapper.to_standard(var_name)
                        if mapped_name in state:
                            state_var_name = mapped_name
                        else:
                            raise KeyError(f"Required input variable not found: {var_name}")
                
                value = state[state_var_name]
                if isinstance(value, Tensor):
                    if value.dim() == 0:
                        column = value.reshape(1)
                    elif value.dim() == 1:
                        column = value
                    else:
                        raise ValueError(
                            f"Surrogate input '{state_var_name}' must be rank-0 or "
                            "rank-1; explicit spatial aggregation belongs to a "
                            "ScaleCouplingFunctor"
                        )
                else:
                    scalar_value = value.item() if hasattr(value, "item") else value
                    column = torch.as_tensor(
                        float(scalar_value),
                        dtype=torch.float32,
                        device=self.device,
                    ).reshape(1)
                
                if var_name in self.units_conversion:
                    from_unit, to_unit = self.units_conversion[var_name]
                    column = self._convert_units(column, from_unit, to_unit)
                    if not isinstance(column, Tensor):
                        column = torch.as_tensor(column)

                input_columns.append(
                    column.to(device=self.device, dtype=torch.float32).reshape(-1)
                )

            if not input_columns:
                return torch.empty((1, 0), dtype=torch.float32, device=self.device)

            batch_size = max(column.numel() for column in input_columns)
            normalized_columns = []
            for var_name, column in zip(self.input_variables, input_columns):
                if column.numel() == 1 and batch_size > 1:
                    column = column.expand(batch_size)
                elif column.numel() != batch_size:
                    raise ValueError(
                        f"Inconsistent batch size for {var_name}: "
                        f"{column.numel()} values, expected 1 or {batch_size}"
                    )
                normalized_columns.append(column)

            return torch.stack(normalized_columns, dim=1)
            
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            # Re-raise validation errors directly for testing consistency
            raise e
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            raise SurrogateError(f"Failed to prepare surrogate input: {e}") from e
    
    def _format_output(self, predictions: Tensor, dt: float) -> FluxDict:
        """Format model predictions as FluxDict.
        
        Parameters:
            predictions : Tensor
                Model predictions, shape [1, n_outputs] or [n_outputs]
            dt : float
                Time step size for rate scaling
                
        Returns:
            FluxDict: Formatted flux dictionary with proper units
        """
        try:
            # Handle batch dimension - predictions can be [batch_size, n_outputs] or [n_outputs]
            if predictions.dim() == 1:
                # Single sample: [n_outputs]
                n_outputs = predictions.shape[0]
            elif predictions.dim() == 2:
                # Batch: [batch_size, n_outputs]
                n_outputs = predictions.shape[1]
            else:
                raise SurrogateError(f"Invalid prediction tensor dimensions: {predictions.shape}")
            
            # Ensure we have the right number of outputs
            if n_outputs != len(self.output_variables):
                raise SurrogateError(
                    f"Model output dimension mismatch: got {n_outputs}, "
                    f"expected {len(self.output_variables)}"
                )
            
            flux_dict = {}
            for i, var_name in enumerate(self.output_variables):
                if predictions.dim() == 1:
                    output_tensor = predictions[i].reshape(1)
                else:
                    output_tensor = predictions[:, i]

                output_var_name = self.output_mapping.get(var_name, var_name)
                if var_name in self.units_conversion:
                    from_unit, to_unit = self.units_conversion[var_name]
                    output_tensor = self._convert_units(
                        output_tensor,
                        from_unit,
                        to_unit,
                    )
                    if not isinstance(output_tensor, Tensor):
                        output_tensor = torch.as_tensor(output_tensor)

                if (
                    self.scale_rate_outputs_by_dt
                    and var_name in self.rate_output_variables
                ):
                    dt_tensor = torch.as_tensor(
                        dt,
                        dtype=output_tensor.dtype,
                        device=output_tensor.device,
                    )
                    output_tensor = output_tensor * dt_tensor

                output_tensor = output_tensor.to(
                    device=self.device,
                    dtype=predictions.dtype,
                )
                if self.output_update_mode is not None:
                    flux_dict[output_var_name] = UpdateItem(
                        value=output_tensor,
                        mode=self.output_update_mode,
                        per_timestep=False,
                        source_functor=self.name,
                    )
                else:
                    flux_dict[output_var_name] = output_tensor
            
            return flux_dict
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            raise SurrogateError(f"Failed to format surrogate output: {e}") from e
    
    def _convert_units(
        self,
        value: Union[Tensor, float],
        from_unit: str,
        to_unit: str,
    ) -> Union[Tensor, float]:
        """Convert value between units using authoritative unit converter.
        
        Uses the RELIC unit conversion system for accurate, science-grade
        explicit unit conversions with bounded CF variable metadata.
        
        Parameters:
            value : float
                Value to convert
            from_unit : str
                Source unit
            to_unit : str
                Target unit
                
        Returns:
            float: Converted value
        """
        # Create conversion context
        context = ConversionContext(allow_identity=True)

        from_unit = self._normalize_unit_alias(from_unit)
        to_unit = self._normalize_unit_alias(to_unit)
        variable_name = 'generic'

        try:
            # Convert using authoritative system
            return unit_converter.convert(value, variable_name, from_unit, to_unit, context)
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as exc:
            raise SurrogateError(
                f"Unit conversion failed: {from_unit} -> {to_unit}"
            ) from exc

    @staticmethod
    def _normalize_unit_alias(unit: str) -> str:
        """Normalize common unit aliases to CF-style units."""
        if unit is None:
            return unit
        unit_norm = unit.strip()
        unit_key = unit_norm.lower()
        aliases = {
            "kelvin": "K",
            "degc": "degC",
            "celsius": "degC",
            "umol_m2_s1": "umol m-2 s-1",
            "mol_m2_s1": "mol m-2 s-1",
            "w_m2": "W m-2",
            "w/m2": "W m-2",
        }
        return aliases.get(unit_key, unit_norm)

    def _check_conservation(self, outputs: FluxDict) -> Dict[str, Tensor]:
        """Check conservation laws for surrogate outputs.
        
        Parameters:
            outputs : FluxDict
                Surrogate model outputs
                
        Returns:
            Dict[str, Tensor]: Conservation violations by law name
        """
        violations: Dict[str, Tensor] = {}
        
        if not self.conservation_check:
            return violations
        
        if self._conservation_matrix is None and not self.conservation_functions:
            raise SurrogateError(
                "conservation_check requires conservation_matrix or an explicit "
                "conservation callback"
            )

        # Apply explicitly registered conservation functions.
        for i, conservation_func in enumerate(self.conservation_functions):
            try:
                violation = torch.as_tensor(
                    conservation_func(outputs),
                    dtype=torch.float64,
                    device=self.device,
                )
            except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
                raise SurrogateError(
                    f"Conservation callback {i} failed: {e}"
                ) from e
            if violation.numel() == 0 or not bool(torch.isfinite(violation).all().item()):
                raise SurrogateError(
                    f"Conservation callback {i} must return finite values"
                )
            violations[f'conservation_{i}'] = violation.abs().amax()

        if self._conservation_matrix is not None:
            columns: List[Tensor] = []
            batch_size: Optional[int] = None
            for variable in self.output_variables:
                runtime_variable = self.output_mapping.get(variable, variable)
                if runtime_variable not in outputs:
                    raise SurrogateError(
                        f"Conservation ledger output is missing: {runtime_variable}"
                    )
                value = outputs[runtime_variable]
                if isinstance(value, UpdateItem):
                    value = value.value
                column = torch.as_tensor(value, device=self.device)
                if column.ndim == 0:
                    column = column.reshape(1)
                elif column.ndim != 1:
                    raise SurrogateError(
                        "Conservation ledger outputs must be rank-0 or rank-1"
                    )
                if batch_size is None:
                    batch_size = column.numel()
                elif column.numel() != batch_size:
                    raise SurrogateError(
                        "Conservation ledger outputs must have a common batch size"
                    )
                columns.append(column)

            output_matrix = torch.stack(columns, dim=1)
            ledger = self._conservation_matrix.to(
                device=output_matrix.device,
                dtype=output_matrix.dtype,
            )

            # References: Karniadakis et al. (2021), Nat. Rev. Phys. 3, 422-440.
            # Mathematical expression: R = Y C^T; require max(abs(R)) <= epsilon.
            # Parameters: Y is the surrogate output matrix, C is the declared
            # conservation ledger, R is the residual, and epsilon is the tolerance.
            residual = output_matrix @ ledger.transpose(0, 1)
            row_violations = residual.abs().amax(dim=0)
            for index, violation in enumerate(row_violations):
                violations[f'ledger_{index}'] = violation
        
        return violations
    
    def add_conservation_function(self, func: ConservationFunction, name: str = None) -> None:
        """Add custom conservation checking function.
        
        Parameters:
            func : ConservationFunction
                Function that takes FluxDict and returns violation magnitude
            name : str, optional
                Name for logging and debugging
        """
        self.conservation_functions.append(func)
        if name:
            logger.info(f"Added conservation function: {name}")
    
    def forward(self, 
                state: StateDict, 
                time: float, 
                dt: float, 
                **kwargs) -> FluxDict:
        """Forward pass for surrogate model inference.
        
        Implements the core FunctorBase interface with surrogate-specific logic:
        1. Prepare inputs from state dictionary
        2. Run surrogate model inference
        3. Format outputs as FluxDict with proper units
        4. Check conservation laws and constraints
        5. Estimate uncertainty if enabled
        
        Parameters:
            state : StateDict
                Current system state variables
            time : float
                Current simulation time [s]
            dt : float
                Time step size [s]
            **kwargs
                Additional context (environment, forcing, etc.)
                
        Returns:
            FluxDict: Process rates and state updates
            
        Raises:
            SurrogateError: If inference fails
            ConservationViolationError: If conservation laws violated
        """
        try:
            if not self.model_loaded:
                raise SurrogateError(f"Surrogate model not loaded: {self.name}")
            
            # Prepare model inputs
            input_tensor = self._prepare_input(state)
            
            # Run inference
            predictions = self._predict(input_tensor)
            
            # Format outputs
            outputs = self._format_output(predictions, dt)
            
            # Check conservation laws
            if self.conservation_check:
                violations = self._check_conservation(outputs)
                
                violation_values = [
                    value.to(device=self.device, dtype=torch.float64).abs().amax()
                    for value in violations.values()
                ]
                max_violation = torch.stack(violation_values).amax()
                tolerance = torch.as_tensor(
                    self.conservation_tolerance,
                    dtype=max_violation.dtype,
                    device=max_violation.device,
                )
                if bool((max_violation > tolerance).item()):
                    violation_value = float(max_violation.detach().cpu().item())
                    error_msg = (
                        f"Conservation violation in {self.name}: {violation_value:.2e} > "
                        f"tolerance {self.conservation_tolerance:.2e}"
                    )
                    raise ConservationViolationError(error_msg)
            
            # Add uncertainty estimates if enabled. A requested estimate that
            # the surrogate cannot provide is an error, never silently omitted.
            if self.uncertainty_quantification:
                uncertainty = self._estimate_uncertainty(input_tensor)
                if uncertainty is None:
                    raise NotImplementedError(
                        f"uncertainty_quantification is enabled but "
                        f"{type(self).__name__} provides no uncertainty estimate"
                    )
                for i, var_name in enumerate(self.output_variables):
                    output_var_name = self.output_mapping.get(var_name, var_name)
                    uncertainty_key = f"{output_var_name}_uncertainty"
                    outputs[uncertainty_key] = uncertainty[0, i] if uncertainty.dim() > 1 else uncertainty[i]
            
            return outputs
            
        except (ImportError, OSError) as e:
            # Backend loading and I/O failures during inference become a
            # SurrogateError with their cause; every other error propagates.
            raise SurrogateError(f"Surrogate inference failed: {e}") from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get surrogate model metadata and statistics.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'model_path': str(self.model_path),
            'model_loaded': self.model_loaded,
            'input_variables': self.input_variables,
            'output_variables': self.output_variables,
            'device': str(self.device),
            'conservation_check': self.conservation_check,
            'conservation_tolerance': self.conservation_tolerance,
            'conservation_matrix_shape': (
                None
                if self._conservation_matrix is None
                else tuple(self._conservation_matrix.shape)
            ),
            'scale_rate_outputs_by_dt': self.scale_rate_outputs_by_dt,
            'rate_output_variables': list(self.rate_output_variables),
            'uncertainty_quantification': self.uncertainty_quantification,
            'batch_size': self.batch_size
        }


class ONNXSurrogate(SurrogateFunctor):
    """
    ONNX Runtime surrogate model implementation.
    
    Provides efficient cross-platform deployment of neural network surrogate
    models using ONNX (Open Neural Network Exchange) format. Supports CPU
    and GPU inference with hardware-specific optimizations.
    
    Key Features:
        - Cross-platform deployment (Windows/Linux/macOS)
        - Hardware optimization (CPU/GPU/TPU) 
        - Quantization support for reduced memory/faster inference
        - Batch processing capabilities
        - Thread-safe inference sessions
    
    ONNX Model Requirements:
        - Input shape: [batch_size, n_features]
        - Output shape: [batch_size, n_outputs]  
        - Float32 precision (recommended)
        - Single input/output tensor (no named I/O)
    
    Additional Config Options:
        providers : List[str], optional
            ONNX execution providers: ['CPUExecutionProvider', 'CUDAExecutionProvider']
            Default: auto-detected based on device
        session_options : Dict, optional
            ONNX Runtime session configuration
        model_metadata : Dict, optional
            Pre-loaded model metadata to avoid file parsing
            
    Performance Notes:
        - First inference may be slower due to optimization
        - GPU inference requires onnxruntime-gpu package  
        - Batch processing improves GPU utilization
        - Model quantization can reduce memory by 4x with minimal accuracy loss
    
    References:
        - ONNX Runtime documentation: https://onnxruntime.ai/
        - Bai, J. et al. (2021). ONNX Runtime graph optimizations
    """
    
    def __init__(self, 
                 name: str,
                 config: ConfigDict,
                 metadata: Optional[FunctorMetadata] = None,
                 **kwargs) -> None:
        """Initialize ONNX surrogate with runtime-specific settings."""
        
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX Runtime not available. Install with: "
                "pip install onnxruntime  # or onnxruntime-gpu for GPU support"
            )
        
        # ONNX-specific configuration
        settings = coerce_config_dict(config)
        if settings.get('uncertainty_quantification', False):
            raise NotImplementedError(
                "uncertainty_quantification is not implemented for ONNX "
                "surrogates (a deterministic ONNX model has no uncertainty "
                "estimate)"
            )
        self.providers: List[str] = settings.get('providers', [])
        self.session_options: Dict = settings.get('session_options', {})
        self.model_metadata: Optional[Dict] = settings.get('model_metadata', None)
        
        # ONNX Runtime components
        self.ort_session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.input_shape: Optional[Tuple[int, ...]] = None
        self.output_shape: Optional[Tuple[int, ...]] = None
        
        super().__init__(name, config, metadata, **kwargs)
        
        logger.info(f"ONNX surrogate initialized: {name} on {self.providers}")
    
    def _load_model(self) -> None:
        """Load ONNX model and create inference session."""
        try:
            # Configure execution providers
            if not self.providers:
                self.providers = self._get_default_providers()
            
            # Create session options
            sess_options = ort.SessionOptions()
            
            # Apply custom session options
            for key, value in self.session_options.items():
                if not hasattr(sess_options, key):
                    raise ValueError(f"Unknown ONNX session option: {key}")
                setattr(sess_options, key, value)
            
            # Create inference session
            self.ort_session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=self.providers
            )
            
            # Get model input/output metadata
            self._extract_model_metadata()
            
            # Verify model consistency
            self._verify_model_consistency()
            
            self.model_loaded = True
            
            logger.info(
                f"ONNX model loaded: {self.model_path} "
                f"(input: {self.input_shape}, output: {self.output_shape})"
            )
            
        except Exception as e:  # noqa: BLE001 - wrap backend-specific ONNX errors as ModelLoadError
            raise ModelLoadError(f"Failed to load ONNX model {self.model_path}: {e}") from e
    
    def _get_default_providers(self) -> List[str]:
        """Get default execution providers based on device preference."""
        available_providers = ort.get_available_providers()
        
        preference = str(self.device_preference).strip().lower()
        if preference.startswith('cuda'):
            # An explicitly requested CUDA device is honoured or refused; it
            # never runs on the CPU provider.
            if 'CUDAExecutionProvider' not in available_providers:
                raise RuntimeError(
                    "CUDA device requested but onnxruntime provides no "
                    "CUDAExecutionProvider"
                )
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if self.device.type == 'cuda' and 'CUDAExecutionProvider' in available_providers:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif self.device.type == 'cpu':
            return ['CPUExecutionProvider']
        else:
            # device = auto (or a non-CPU device without a matching provider):
            # the documented automatic choice of the best available providers.
            preferred_order = [
                'CUDAExecutionProvider',    # GPU (NVIDIA) 
                'ROCMExecutionProvider',    # GPU (AMD)
                'CoreMLExecutionProvider',  # Apple Silicon
                'CPUExecutionProvider'      # CPU provider
            ]
            
            providers = []
            for provider in preferred_order:
                if provider in available_providers:
                    providers.append(provider)
            
            return providers[:2]  # Use top 2 in priority order
    
    def _extract_model_metadata(self) -> None:
        """Extract input/output metadata from ONNX model."""
        if self.ort_session is None:
            raise RuntimeError("ONNX session not initialized")
        
        # Get input metadata
        inputs = self.ort_session.get_inputs()
        if len(inputs) != 1:
            raise ModelLoadError(f"Expected 1 input, got {len(inputs)}")
        
        input_meta = inputs[0]
        self.input_name = input_meta.name
        self.input_shape = tuple(input_meta.shape)
        
        # Get output metadata
        outputs = self.ort_session.get_outputs()
        if len(outputs) != 1:
            raise ModelLoadError(f"Expected 1 output, got {len(outputs)}")
        
        output_meta = outputs[0]
        self.output_name = output_meta.name
        self.output_shape = tuple(output_meta.shape)
        
        logger.debug(
            f"ONNX I/O: {self.input_name}{self.input_shape} -> "
            f"{self.output_name}{self.output_shape}"
        )
    
    def _verify_model_consistency(self) -> None:
        """Verify ONNX model is consistent with surrogate interface."""
        if self.input_shape is None or self.output_shape is None:
            raise ModelLoadError("Model metadata not extracted")
        
        # Check input dimensions
        if len(self.input_shape) != 2:
            raise ModelLoadError(
                f"Expected 2D input [batch, features], got shape {self.input_shape}"
            )
        
        # Check feature dimension matches expected inputs
        n_features = self.input_shape[1] if self.input_shape[1] != -1 else len(self.input_variables)
        if n_features != len(self.input_variables):
            raise ModelLoadError(
                f"ONNX input feature mismatch: model expects {n_features}, "
                f"config specifies {len(self.input_variables)}"
            )
        
        # Check output dimensions
        if len(self.output_shape) != 2:
            raise ModelLoadError(
                f"Expected 2D output [batch, outputs], got shape {self.output_shape}"
            )
        
        # Check output dimension matches expected outputs
        n_outputs = self.output_shape[1] if self.output_shape[1] != -1 else len(self.output_variables)
        if n_outputs != len(self.output_variables):
            raise ModelLoadError(
                f"ONNX output dimension mismatch: model produces {n_outputs}, "
                f"config specifies {len(self.output_variables)}"
            )
    
    def _predict(self, input_tensor: Tensor) -> Tensor:
        """Run ONNX model inference.
        
        Parameters:
            input_tensor : Tensor
                Model inputs, shape [batch_size, n_features]
                
        Returns:
            Tensor: Model outputs, shape [batch_size, n_outputs]
        """
        try:
            if self.ort_session is None:
                raise InferenceError("ONNX session not initialized")
            
            # Convert to numpy for ONNX Runtime
            input_np = input_tensor.detach().cpu().numpy()
            
            # Prepare input dictionary
            ort_inputs = {self.input_name: input_np}
            
            # Run inference
            ort_outputs = self.ort_session.run(None, ort_inputs)
            
            # Convert back to PyTorch tensor
            output_np = ort_outputs[0]
            output_tensor = torch.from_numpy(output_np).to(
                device=self.device,
                dtype=torch.float32
            )
            
            return output_tensor
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            raise InferenceError(f"ONNX inference failed: {e}") from e
    
    def run_batch(self, input_batch: Tensor) -> Tensor:
        """Run batch inference for multiple inputs.
        
        Parameters:
            input_batch : Tensor
                Batch of inputs, shape [batch_size, n_features]
                
        Returns:
            Tensor: Batch of outputs, shape [batch_size, n_outputs]
        """
        if input_batch.size(0) <= self.batch_size:
            # Single batch
            return self._predict(input_batch)
        else:
            # Split into smaller batches
            outputs = []
            for i in range(0, input_batch.size(0), self.batch_size):
                batch_slice = input_batch[i:i+self.batch_size]
                batch_output = self._predict(batch_slice)
                outputs.append(batch_output)
            
            return torch.cat(outputs, dim=0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get ONNX model information."""
        base_info = super().get_model_info()
        
        onnx_info = {
            'providers': self.providers,
            'input_name': self.input_name,
            'output_name': self.output_name,
            'input_shape': self.input_shape,
            'output_shape': self.output_shape,
            'session_options': self.session_options
        }
        
        return {**base_info, **onnx_info}
    
    def benchmark_performance(self, n_iterations: int = 100) -> Dict[str, float]:
        """Benchmark ONNX model inference performance.
        
        Parameters:
            n_iterations : int
                Number of inference iterations
                
        Returns:
            Dict[str, float]: Performance statistics
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")
        
        # Create dummy input
        dummy_input = torch.randn(1, len(self.input_variables), device=self.device)
        
        # Warmup
        for _ in range(10):
            self._predict(dummy_input)
        
        # Benchmark
        import time
        
        start_time = time.perf_counter()
        
        for _ in range(n_iterations):
            self._predict(dummy_input)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time = total_time / n_iterations
        throughput = n_iterations / total_time
        
        return {
            'total_time_s': total_time,
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_inferences_per_s': throughput,
            'iterations': n_iterations
        }


PYTORCH_SURROGATE_AVAILABLE = False
_PYTORCH_SURROGATE_IMPORT_ERROR = None
_LAZY_PYTORCH_EXPORTS = frozenset({"PyTorchSurrogate", "SurrogateDataset"})

# Export public API
__all__ = [
    'SurrogateFunctor',
    'ONNXSurrogate', 
    'SurrogateError',
    'ModelLoadError',
    'InferenceError',
    'ConservationViolationError',
    'VariableMapping',
    'ConservationFunction',
    'UncertaintyEstimator'
]

if importlib.util.find_spec("relic.models.pytorch_surrogate") is not None:
    __all__.extend(['PyTorchSurrogate', 'SurrogateDataset'])


def _ensure_pytorch_surrogate_loaded() -> bool:
    """Load PyTorch surrogate symbols lazily to avoid import cycles."""
    global PYTORCH_SURROGATE_AVAILABLE
    global _PYTORCH_SURROGATE_IMPORT_ERROR

    if PYTORCH_SURROGATE_AVAILABLE:
        return True

    if importlib.util.find_spec("relic.models.pytorch_surrogate") is None:
        _PYTORCH_SURROGATE_IMPORT_ERROR = ModuleNotFoundError(
            "relic.models.pytorch_surrogate is not available"
        )
        return False
    pytorch_module = importlib.import_module("relic.models.pytorch_surrogate")
    PyTorchSurrogate = getattr(pytorch_module, "PyTorchSurrogate")
    SurrogateDataset = getattr(pytorch_module, "SurrogateDataset")

    globals().update(
        {
            "PyTorchSurrogate": PyTorchSurrogate,
            "SurrogateDataset": SurrogateDataset,
        }
    )
    PYTORCH_SURROGATE_AVAILABLE = True
    _PYTORCH_SURROGATE_IMPORT_ERROR = None
    return True


def __getattr__(name: str):
    if name in _LAZY_PYTORCH_EXPORTS:
        if _ensure_pytorch_surrogate_loaded():
            return globals()[name]
        raise AttributeError(
            f"PyTorch surrogate is unavailable: {_PYTORCH_SURROGATE_IMPORT_ERROR}"
        )
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
