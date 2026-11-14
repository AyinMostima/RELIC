"""
PyTorch Native Surrogate Model Implementation

This module provides native PyTorch surrogate models with full training
and inference capabilities, supporting gradient-based optimization and
automatic differentiation throughout the modeling pipeline.

Key Features:
    - Native PyTorch nn.Module integration
    - Online/offline training with backpropagation
    - Automatic gradient computation for optimization
    - Model checkpointing and restoration
    - Ensemble support for uncertainty quantification
    - Physics-informed neural network capabilities

Mathematical Foundation:
    Training objective: min_theta L(y, y_hat) + lambdaR(theta)
    where:
        L: Loss function (MSE, MAE, custom physics loss)
        R: Regularization term (L1, L2, physics constraints)
        theta: Model parameters
        lambda: Regularization weight
    
    Gradient update: theta_{t+1} = theta_t - etanabla_thetaL
    where eta is the learning rate

References:
    - Goodfellow, I. et al. (2016). Deep Learning
    - Raissi, M. et al. (2019). Physics-informed neural networks
    - Lagergren, J.H. et al. (2020). Learning PDE solution operators

Author: Yuchong Long
Date: 2025-09-06
"""

import logging
import pickle
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split

# Import RELIC components
from .surrogates import (
    SurrogateFunctor, InferenceError, ModelLoadError
)
from ..core.functor import FunctorMetadata, ConfigDict
from ..core.abstractions import StateDict, FluxDict
from ..core.units import UnitSystem
from ..data.standards.unit_converter import unit_converter, ConversionContext
from .config_utils import coerce_config_dict

logger = logging.getLogger(__name__)

# Prediction modes of PyTorchSurrogate (config key uncertainty.method).
# 'none' is the deterministic forward pass; both other modes sample dropout
# masks. Any other name is rejected at construction.
UNCERTAINTY_METHODS = ("none", "dropout_mc", "ensemble")

_DROPOUT_MODULES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
    nn.FeatureAlphaDropout,
)


def _initialize_dataloader_worker(worker_id: int) -> None:
    """Initialize one picklable DataLoader RNG substream on Windows spawn."""
    _ = worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    init_threads = getattr(torch, "init_num_threads", None)
    if callable(init_threads):
        init_threads()


class SurrogateDataset(Dataset):
    """Training dataset with explicit, validated unit declarations."""

    def __init__(
        self,
        states: List[StateDict],
        fluxes: List[FluxDict],
        input_variables: List[str],
        output_variables: List[str],
        input_units: Optional[Dict[str, str]] = None,
        output_units: Optional[Dict[str, str]] = None,
        unit_conversions: Optional[Dict[str, Tuple[str, str]]] = None,
        validate_units: bool = True,
    ) -> None:
        """Initialize a dataset without inferring units from variable names."""
        self.states = states
        self.fluxes = fluxes
        self.input_variables = list(input_variables)
        self.output_variables = list(output_variables)
        self.input_units = dict(input_units or {})
        self.output_units = dict(output_units or {})
        self.unit_conversions = dict(unit_conversions or {})
        self.validate_units = validate_units

        if len(states) != len(fluxes):
            raise ValueError(
                f"States ({len(states)}) and fluxes ({len(fluxes)}) "
                "must have same length"
            )
        if validate_units:
            self._validate_unit_consistency()

    @staticmethod
    def _require_valid_unit(role: str, variable: str, unit: str) -> None:
        """Require a parseable unit declaration."""
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(f"Invalid {role} unit for {variable}: {unit!r}")
        try:
            dimension = unit_converter.get_dimension(unit)
        except (
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            RuntimeError,
            ImportError,
            OSError,
        ) as exc:
            raise ValueError(
                f"Invalid {role} unit for {variable}: {unit}"
            ) from exc
        if dimension is None:
            raise ValueError(f"Invalid {role} unit for {variable}: {unit}")

    def _validate_unit_consistency(self) -> None:
        """Validate declarations and explicit conversion directions."""
        input_names = set(self.input_variables)
        output_names = set(self.output_variables)
        missing_inputs = sorted(input_names - set(self.input_units))
        missing_outputs = sorted(output_names - set(self.output_units))
        extra_inputs = sorted(set(self.input_units) - input_names)
        extra_outputs = sorted(set(self.output_units) - output_names)
        if missing_inputs:
            raise ValueError(f"Missing input unit declarations: {missing_inputs}")
        if missing_outputs:
            raise ValueError(f"Missing output unit declarations: {missing_outputs}")
        if extra_inputs:
            raise ValueError(f"Unexpected input unit declarations: {extra_inputs}")
        if extra_outputs:
            raise ValueError(f"Unexpected output unit declarations: {extra_outputs}")

        for variable, unit in self.input_units.items():
            self._require_valid_unit("input", variable, unit)
        for variable, unit in self.output_units.items():
            self._require_valid_unit("output", variable, unit)

        declared_names = input_names | output_names
        extra_conversions = sorted(set(self.unit_conversions) - declared_names)
        if extra_conversions:
            raise ValueError(
                f"Unit conversions contain undeclared variables: {extra_conversions}"
            )
        for variable, conversion in self.unit_conversions.items():
            if (
                not isinstance(conversion, (tuple, list))
                or len(conversion) != 2
            ):
                raise ValueError(
                    f"Unit conversion for {variable} must be a two-item pair"
                )
            source_unit, target_unit = conversion
            self._require_valid_unit("conversion source", variable, source_unit)
            self._require_valid_unit("conversion target", variable, target_unit)
            if (
                variable in input_names
                and self.input_units[variable] != source_unit
            ):
                raise ValueError(
                    f"Input unit for {variable} must match conversion source "
                    f"{source_unit}"
                )
            if (
                variable in output_names
                and self.output_units[variable] != target_unit
            ):
                raise ValueError(
                    f"Output unit for {variable} must match conversion target "
                    f"{target_unit}"
                )

    @staticmethod
    def _extract_scalar(value: Any, variable: str, role: str) -> float:
        """Extract one scalar sample without implicit spatial reduction."""
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"{role.capitalize()} variable {variable} must be scalar "
                    f"per training sample, got shape {tuple(value.shape)}"
                )
            return float(value.item())
        return float(value)

    def _convert_training_value(
        self,
        value: float,
        variable: str,
        *,
        role: str,
    ) -> float:
        """Apply only the configured runtime conversion to training data."""
        conversion = self.unit_conversions.get(variable)
        if not self.validate_units or conversion is None:
            return value
        source_unit, target_unit = conversion
        if role == "output":
            source_unit, target_unit = target_unit, source_unit
        try:
            converted = unit_converter.convert(
                value,
                variable,
                source_unit,
                target_unit,
                ConversionContext(allow_identity=True),
            )
        except (
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            RuntimeError,
            ImportError,
            OSError,
        ) as exc:
            raise ValueError(
                f"{role.capitalize()} unit conversion failed for {variable}: "
                f"{source_unit} -> {target_unit}"
            ) from exc
        return float(converted)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """Return one explicitly validated model-space training sample."""
        state = self.states[idx]
        flux = self.fluxes[idx]

        input_values: List[float] = []
        for variable in self.input_variables:
            if variable not in state:
                raise KeyError(f"Variable {variable} not found in state")
            value = self._extract_scalar(state[variable], variable, "input")
            input_values.append(
                self._convert_training_value(value, variable, role="input")
            )

        output_values: List[float] = []
        for variable in self.output_variables:
            if variable not in flux:
                raise KeyError(f"Variable {variable} not found in flux")
            value = self._extract_scalar(flux[variable], variable, "output")
            output_values.append(
                self._convert_training_value(value, variable, role="output")
            )

        return (
            torch.tensor(input_values, dtype=torch.float32),
            torch.tensor(output_values, dtype=torch.float32),
        )


class BasicResNetBlock(nn.Module):
    """Basic ResNet block with skip connections.
    
    Mathematical Expression:
        y = F(x) + x  (where F(x) is the residual function)
        F(x) = Conv(ReLU(BN(Conv(x))))
    
    References:
        - He, K. et al. (2015). Deep Residual Learning for Image Recognition
    """
    
    def __init__(self, dim: int, activation: str, dropout_rate: float):
        super().__init__()
        self.conv1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        if self.dropout:
            out = self.dropout(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += identity
        out = self.activation(out)
        
        return out


class SimpleTransformerBlock(nn.Module):
    """Simplified Transformer block with linear attention.
    
    Mathematical Expression:
        Attention(Q,K,V) = softmax(QK^T/sqrtd_k)V
        Output = LayerNorm(x + MultiHeadAttention(x)) + FFN(x)
    
    References:
        - Vaswani, A. et al. (2017). Attention Is All You Need
    """
    
    def __init__(self, d_model: int, n_heads: int, ffn_ratio: float,
                 dropout_rate: float):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model) 
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Feed-forward network
        ffn_dim = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through transformer block.
        
        Parameters:
            x : Tensor, shape [batch_size, seq_len, d_model] or [batch_size, d_model]
        """
        # Handle 2D input (batch_size, d_model) by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, d_model]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len, d_model = x.shape
        
        # Self-attention
        residual = x
        x = self.ln1(x)
        
        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        if self.dropout:
            attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        attn_output = self.out_proj(attn_output)
        
        # Add residual
        x = residual + attn_output
        
        # Feed-forward network
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x
        
        # Remove sequence dimension if it was added
        if squeeze_output:
            x = x.squeeze(1)
        
        return x


def architecture_factory(arch_config: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
    """Factory function to create different neural network architectures.
    
    Parameters:
        arch_config : Dict[str, Any]
            Architecture configuration containing 'type' and type-specific parameters
        input_dim : int
            Input feature dimension
        output_dim : int
            Output feature dimension
            
    Returns:
        nn.Module: Configured neural network architecture
    """
    arch_settings = resolve_architecture(arch_config)
    arch_type = arch_settings['type']

    if arch_type == 'mlp':
        return MLPSurrogateNet(input_dim, output_dim, arch_settings)
    elif arch_type == 'resnet':
        return ResNetSurrogateNet(input_dim, output_dim, arch_settings)
    elif arch_type == 'transformer':
        return TransformerSurrogateNet(input_dim, output_dim, arch_settings)
    else:
        raise ValueError(f"Unknown architecture type: {arch_type}")


_ARCHITECTURE_KEYS = {
    'mlp': frozenset({'type', 'hidden_dims', 'activation', 'dropout_rate', 'batch_norm'}),
    'resnet': frozenset({'type', 'n_blocks', 'hidden_dim', 'activation', 'dropout_rate'}),
    'transformer': frozenset({'type', 'd_model', 'n_heads', 'n_layers', 'ffn_ratio', 'dropout_rate'}),
}
_HYPERPARAMETER_PREFIX = "models.surrogate.pytorch"


def resolve_architecture(arch_config: Any) -> Dict[str, Any]:
    """Complete an architecture configuration from the YAML database.

    The configured keys are kept; the network type (when absent) and every
    key the chosen network reads but the configuration omits are read from
    models.surrogate.pytorch.architecture.* (models/pytorch_surrogate.yaml).
    A key the chosen network does not read (a typo such as hidden_dim for an
    MLP, or batch_norm for the ResNet whose blocks always normalize) is
    rejected instead of being ignored. There is no code default.
    """
    from relic.parameters import get_global_manager

    manager = get_global_manager()
    settings = dict(coerce_config_dict(arch_config))
    if 'type' not in settings:
        settings['type'] = manager.get(f"{_HYPERPARAMETER_PREFIX}.architecture.type", required=True)
    arch_type = str(settings['type']).lower()
    if arch_type not in _ARCHITECTURE_KEYS:
        raise ValueError(f"Unknown architecture type: {arch_type}")
    settings['type'] = arch_type
    unread = sorted(set(settings) - _ARCHITECTURE_KEYS[arch_type])
    if unread:
        raise ValueError(
            f"architecture keys {unread} are not read by the {arch_type} network; "
            f"accepted keys: {sorted(_ARCHITECTURE_KEYS[arch_type])}"
        )
    for key in sorted(_ARCHITECTURE_KEYS[arch_type] - set(settings)):
        settings[key] = manager.get(
            f"{_HYPERPARAMETER_PREFIX}.architecture.{arch_type}.{key}", required=True
        )
    return settings


class MLPSurrogateNet(nn.Module):
    """Multi-Layer Perceptron architecture (original FlexibleSurrogateNet logic)."""
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super().__init__()
        
        # Extract MLP-specific parameters
        settings = resolve_architecture({**coerce_config_dict(config), 'type': 'mlp'})
        hidden_dims = settings['hidden_dims']
        activation = settings['activation']
        dropout_rate = settings['dropout_rate']
        batch_norm = settings['batch_norm']
        if not isinstance(batch_norm, bool):
            raise TypeError(f"architecture.batch_norm must be a boolean, got {batch_norm!r}")
        
        # Build layers (original FlexibleSurrogateNet logic)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Add activation
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Store architecture info
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
    
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class ResNetSurrogateNet(nn.Module):
    """ResNet-based surrogate architecture with skip connections."""
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super().__init__()
        
        # Extract ResNet-specific parameters
        settings = resolve_architecture({**coerce_config_dict(config), 'type': 'resnet'})
        n_blocks = settings['n_blocks']
        hidden_dim = settings['hidden_dim']
        activation = settings['activation']
        dropout_rate = settings['dropout_rate']
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # ResNet blocks
        self.blocks = nn.ModuleList([
            BasicResNetBlock(hidden_dim, activation, dropout_rate)
            for _ in range(n_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Store architecture info
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim
    
    def forward(self, x: Tensor) -> Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Pass through ResNet blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class TransformerSurrogateNet(nn.Module):
    """Transformer-based surrogate architecture with self-attention."""
    
    def __init__(self, input_dim: int, output_dim: int, config: Dict[str, Any]):
        super().__init__()
        
        # Extract Transformer-specific parameters
        settings = resolve_architecture({**coerce_config_dict(config), 'type': 'transformer'})
        d_model = settings['d_model']
        n_heads = settings['n_heads']
        n_layers = settings['n_layers']
        ffn_ratio = settings['ffn_ratio']
        dropout_rate = settings['dropout_rate']
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads} for multi-head attention"
            )
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, ffn_ratio, dropout_rate)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, output_dim)
        
        # Store architecture info
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_layers = n_layers
    
    def forward(self, x: Tensor) -> Tensor:
        # Input projection
        x = self.input_proj(x)
        
        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class FlexibleSurrogateNet(nn.Module):
    """Flexible neural network architecture factory for surrogate modeling.
    
    Now serves as a factory that can create MLP, ResNet, or Transformer architectures
    based on configuration.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 architecture: Optional[Dict[str, Any]] = None):
        """Initialize surrogate network using architecture factory.
        
        Parameters:
            input_dim : int
                Number of input features
            output_dim : int
                Number of output features
            architecture : Dict[str, Any], optional
                Architecture configuration for multi-architecture support:
                - type: 'mlp' | 'resnet' | 'transformer' (absent: YAML
                  models.surrogate.pytorch.architecture.type)
                - MLP: hidden_dims, activation, dropout_rate, batch_norm
                - ResNet: n_blocks, hidden_dim, activation, dropout_rate
                - Transformer: d_model, n_heads, n_layers, ffn_ratio, dropout_rate
        
        Examples:
            >>> # Default MLP architecture
            >>> net = FlexibleSurrogateNet(10, 2, architecture={'type': 'mlp'})
            
            >>> # New ResNet architecture
            >>> net = FlexibleSurrogateNet(10, 2, architecture={
            ...     'type': 'resnet', 'n_blocks': 4, 'hidden_dim': 128
            ... })
            
            >>> # New Transformer architecture
            >>> net = FlexibleSurrogateNet(10, 2, architecture={
            ...     'type': 'transformer', 'd_model': 128, 'n_heads': 8
            ... })
        """
        super().__init__()
        
        # Store basic info
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # The architecture is completed from the YAML database (no code default)
        architecture_settings = resolve_architecture(architecture or {})
        self.architecture_config = architecture_settings
        self.architecture_type = architecture_settings["type"]
        self.network = architecture_factory(architecture_settings, input_dim, output_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through network.
        
        Parameters:
            x : Tensor
                Input tensor, shape [batch_size, input_dim]
                
        Returns:
            Tensor: Output tensor, shape [batch_size, output_dim]
        """
        return self.network(x)
    
    def get_architecture_info(self) -> Dict[str, Any]:
        """Get architecture information and configuration.
        
        Returns:
            Dict[str, Any]: Architecture type, configuration, and parameters
        """
        info = {
            'type': self.architecture_type,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'config': self.architecture_config
        }
        
        # Add type-specific information
        if hasattr(self.network, 'hidden_dims'):
            info['hidden_dims'] = self.network.hidden_dims
        if hasattr(self.network, 'n_blocks'):
            info['n_blocks'] = self.network.n_blocks
            info['hidden_dim'] = self.network.hidden_dim
        if hasattr(self.network, 'n_layers'):
            info['n_layers'] = self.network.n_layers
            info['d_model'] = self.network.d_model
        
        return info


def _plain_value(value: Any) -> Any:
    """A 0-d tensor from the parameter system as a Python scalar; other values unchanged."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("a scalar hyperparameter must have one element")
        return value.item()
    return value


class PyTorchSurrogate(SurrogateFunctor):
    """
    Native PyTorch surrogate model with training capabilities.
    
    This class extends SurrogateFunctor to provide full PyTorch nn.Module
    integration, including online training, gradient-based optimization,
    and model persistence.
    
    Key Features:
        - Full backpropagation support for training
        - Multiple optimizer and loss function options
        - Learning rate scheduling
        - Early stopping and model checkpointing
        - Ensemble training for uncertainty quantification
        - Physics-informed loss functions
    
    Additional Config Options:
        architecture : Dict
            Neural network architecture specification:
            - hidden_dims: List[int] (default: [64, 32])
            - activation: str (default: 'relu')
            - dropout_rate: float (default: 0.0)
            - batch_norm: bool (default: False)
        
        training : Dict
            Training configuration:
            - optimizer: str ('adam', 'sgd', 'rmsprop')
            - learning_rate: float (default: 1e-3)
            - batch_size: int (default: 32)
            - epochs: int (default: 100)
            - loss_function: str ('mse', 'mae', 'huber')
            - early_stopping_patience: int (default: 10)
            - validation_split: float, optional (no default): fraction of the
              training data held out for validation (seeded by training.seed)
              when train() gets no validation_data; explicit validation_data
              takes precedence
            - shuffle_training: bool (default: True)
        
        physics_loss : Dict, optional
            Physics-informed loss configuration:
            - conservation_weight: float
            - smoothness_weight: float
            - monotonicity_constraints: List[str]
    
    Example:
        >>> config = {
        ...     'model_path': 'models/photosynthesis.pth',
        ...     'input_variables': ['temperature', 'par', 'co2'],
        ...     'output_variables': ['gpp', 'respiration'],
        ...     'architecture': {
        ...         'hidden_dims': [128, 64, 32],
        ...         'activation': 'relu',
        ...         'dropout_rate': 0.1
        ...     },
        ...     'training': {
        ...         'optimizer': 'adam',
        ...         'learning_rate': 1e-3,
        ...         'epochs': 200,
        ...         'batch_size': 64
        ...     }
        ... }
        >>> surrogate = PyTorchSurrogate('gpp_surrogate', config)
        >>> surrogate.train(training_data)
        >>> outputs = surrogate(state, time, dt)
    """
    
    def __init__(self,
                 name: str,
                 config: ConfigDict,
                 metadata: Optional[FunctorMetadata] = None,
                 **kwargs) -> None:
        """Initialize PyTorch surrogate model."""
        
        # Architecture configuration
        settings = coerce_config_dict(config)
        self.architecture_config = coerce_config_dict(settings.get('architecture'))
        self.training_config = coerce_config_dict(settings.get('training'))
        self.physics_loss_config = coerce_config_dict(settings.get('physics_loss'))
        self.uncertainty_config = coerce_config_dict(settings.get('uncertainty'))
        self.ensemble_config = coerce_config_dict(settings.get('ensemble'))
        # An absent uncertainty section means the deterministic forward pass;
        # a misspelled method is an error, never a deterministic prediction.
        uncertainty_method = self.uncertainty_config.get('method', 'none')
        if uncertainty_method not in UNCERTAINTY_METHODS:
            raise ValueError(
                f"uncertainty.method must be one of {UNCERTAINTY_METHODS}, got "
                f"{uncertainty_method!r}"
            )
        self.uncertainty_method = uncertainty_method
        self.input_units_config = self._coerce_unit_mapping(
            settings.get("input_units"),
            "input_units",
        )
        self.output_units_config = self._coerce_unit_mapping(
            settings.get("output_units"),
            "output_units",
        )
        
        # Model components
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.loss_fn: Optional[nn.Module] = None
        
        # Training state
        self.is_trained: bool = False
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        self.best_model_state: Optional[Dict] = None
        self.epochs_trained: int = 0
        
        # Initialize base class
        super().__init__(name, config, metadata, **kwargs)
        self._validate_unit_configuration()
        
        logger.info(f"PyTorch surrogate initialized: {name}")
    
    def _load_model(self) -> None:
        """Load or create PyTorch model."""
        try:
            if self.model_path.exists():
                # Load existing model
                logger.info(f"Loading existing model from {self.model_path}")
                checkpoint = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=True,
                )
                
                # Create model architecture
                self._create_model()
                
                # Load weights
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.is_trained = True
                    
                    # Load training history if available
                    if 'training_history' in checkpoint:
                        self.training_history = checkpoint['training_history']
                    if 'epochs_trained' in checkpoint:
                        self.epochs_trained = checkpoint['epochs_trained']
                else:
                    # Old format - just weights
                    self.model.load_state_dict(checkpoint)
                    self.is_trained = True
                
                self.model.to(self.device)
                self.model.eval()
                
            else:
                # Create new model
                logger.info(f"Creating new model (no existing file at {self.model_path})")
                self._create_model()
                self.model.to(self.device)
            
            self.model_loaded = True
            
        except (
            pickle.UnpicklingError,
            EOFError,
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            RuntimeError,
            ImportError,
            OSError,
        ) as e:
            # Every load failure is a ModelLoadError with its cause; an
            # untrained network is never substituted (whatever the path name).
            raise ModelLoadError(f"Failed to load PyTorch model: {e}") from e
    
    def _create_model(self) -> None:
        """Create neural network model based on configuration."""
        input_dim = len(self.input_variables)
        output_dim = len(self.output_variables)
        architecture_settings = coerce_config_dict(self.architecture_config)
        self.model = FlexibleSurrogateNet(
            input_dim=input_dim,
            output_dim=output_dim,
            architecture=architecture_settings,
        )
        
        # Create optimizer
        self._create_optimizer()
        
        # Create loss function
        self._create_loss_function()
    
    def _create_optimizer(self) -> None:
        """Create optimizer for training."""
        if self.model is None:
            return
        
        training_settings = self.training_config
        optimizer_name = str(self._training_setting('optimizer')).lower()
        learning_rate = float(self._training_setting('learning_rate'))
        weight_decay = float(self._training_setting('weight_decay'))

        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = float(self._training_setting('momentum'))
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Create learning rate scheduler if specified. A declared scheduler
        # names its type; an unknown or missing type is rejected instead of
        # training with an implied or no scheduler.
        scheduler_settings = coerce_config_dict(training_settings.get('scheduler'))
        if scheduler_settings:
            if 'type' not in scheduler_settings:
                raise ValueError(
                    "training.scheduler.type must be declared: one of 'step', "
                    "'exponential', 'cosine'"
                )
            scheduler_type = scheduler_settings['type']
            
            if scheduler_type == 'step':
                step_size = int(self._scheduler_setting(scheduler_settings, 'step', 'step_size'))
                gamma = float(self._scheduler_setting(scheduler_settings, 'step', 'gamma'))
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=step_size, gamma=gamma
                )
            elif scheduler_type == 'exponential':
                gamma = float(self._scheduler_setting(scheduler_settings, 'exponential', 'gamma'))
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=gamma
                )
            elif scheduler_type == 'cosine':
                T_max = int(self._scheduler_setting(scheduler_settings, 'cosine', 'T_max'))
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=T_max
                )
            else:
                raise ValueError(
                    f"Unknown training.scheduler type '{scheduler_type}'; expected one of "
                    "'step', 'exponential', 'cosine'"
                )
    
    def _training_setting(self, name: str) -> Any:
        """A training hyperparameter: the configured value, else the YAML value.

        Undeclared values are read from models.surrogate.pytorch.training.<name>
        (models/pytorch_surrogate.yaml) through the Functor parameter entry;
        there is no code default.
        """
        if name in self.training_config:
            return self.training_config[name]
        return _plain_value(
            self.get_parameter_value(f"{_HYPERPARAMETER_PREFIX}.training.{name}", required=True)
        )

    def _scheduler_setting(self, scheduler_settings: Dict[str, Any], scheduler_type: str, name: str) -> Any:
        """A scheduler hyperparameter: the configured value, else the YAML value."""
        if name in scheduler_settings:
            return scheduler_settings[name]
        return _plain_value(
            self.get_parameter_value(
                f"{_HYPERPARAMETER_PREFIX}.scheduler.{scheduler_type}.{name}", required=True
            )
        )

    def _create_loss_function(self) -> None:
        """Create loss function for training."""
        loss_name = str(self._training_setting('loss_function')).lower()

        if loss_name == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_name == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss_name == 'huber':
            delta = float(self._training_setting('huber_delta'))
            self.loss_fn = nn.HuberLoss(delta=delta)
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def _predict(self, input_tensor: Tensor) -> Tensor:
        """Run model inference with optional uncertainty quantification.
        
        Parameters:
            input_tensor : Tensor
                Input features, shape [batch_size, input_dim]
                
        Returns:
            Tensor: Model predictions, shape [batch_size, output_dim]
                   If uncertainty enabled, returns mean predictions
        """
        try:
            if self.model is None:
                raise InferenceError("Model not created")
            
            # The method was validated against UNCERTAINTY_METHODS at
            # construction; each one runs as declared.
            if self.uncertainty_method == 'dropout_mc':
                return self._predict_with_dropout_mc(input_tensor)
            if self.uncertainty_method == 'ensemble':
                return self._predict_with_ensemble(input_tensor)
            return self._predict_deterministic(input_tensor)
            
        except (TypeError, ValueError, KeyError, AttributeError, RuntimeError, ImportError, OSError) as e:
            raise InferenceError(f"PyTorch inference failed: {e}") from e

    def _require_active_dropout(self, method: str) -> None:
        """Require a dropout layer with p > 0 for dropout-sampled predictions.

        Without one every sampled forward pass is identical and the reported
        spread would be a zero 'uncertainty'.
        """
        has_active_dropout = any(
            isinstance(module, _DROPOUT_MODULES) and float(module.p) > 0.0
            for module in self.model.modules()
        )
        if not has_active_dropout:
            raise ValueError(
                f"uncertainty.method '{method}' samples dropout masks, but the "
                "network has no dropout layer with p > 0 (set "
                "architecture.dropout_rate)"
            )

    @contextmanager
    def _temporary_batchnorm_eval(self):
        """Temporarily freeze BatchNorm statistics for single-sample forwards."""
        if self.model is None:
            yield
            return

        batchnorm_modules: List[Tuple[nn.Module, bool]] = []
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                batchnorm_modules.append((module, bool(module.training)))
                module.eval()

        try:
            yield
        finally:
            for module, was_training in batchnorm_modules:
                module.train(was_training)

    @contextmanager
    def _dropout_sampling_mode(self):
        """Enable dropout sampling while keeping normalization layers deterministic."""
        if self.model is None:
            raise InferenceError("Model not created")

        module_training_states = {
            module: bool(module.training)
            for module in self.model.modules()
        }
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, _DROPOUT_MODULES):
                module.train(True)

        try:
            yield
        finally:
            for module, was_training in module_training_states.items():
                module.train(was_training)
    
    def _predict_deterministic(self, input_tensor: Tensor) -> Tensor:
        """Standard deterministic prediction."""
        self.model.eval()
        
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            target_dtype = next(self.model.parameters()).dtype
            if input_tensor.dtype != target_dtype:
                input_tensor = input_tensor.to(dtype=target_dtype)
            output = self.model(input_tensor)
        
        return output
    
    def _predict_with_dropout_mc(self, input_tensor: Tensor) -> Tensor:
        """Monte Carlo Dropout for uncertainty estimation.
        
        References:
            - Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation
        """
        self._require_active_dropout('dropout_mc')
        mc_samples = self._require_positive_count(
            self.uncertainty_config, 'mc_samples', 'uncertainty.mc_samples'
        )
        
        input_tensor = input_tensor.to(self.device)
        
        # Collect multiple forward passes
        predictions = []
        
        with self._dropout_sampling_mode():
            with torch.no_grad():
                for _ in range(mc_samples):
                    output = self.model(input_tensor)
                    predictions.append(output)
        
        # Stack predictions and compute statistics
        predictions = torch.stack(predictions, dim=0)  # [mc_samples, batch_size, output_dim]
        
        # Return mean prediction
        mean_prediction = torch.mean(predictions, dim=0)
        
        # Store uncertainty information for later access
        self._last_prediction_uncertainty = {
            'mean': mean_prediction,
            'std': torch.std(predictions, dim=0),
            'samples': predictions,
            'method': 'dropout_mc'
        }
        
        return mean_prediction
    
    def _predict_with_ensemble(self, input_tensor: Tensor) -> Tensor:
        """Lightweight ensemble prediction for uncertainty.
        
        Note: This is a simplified ensemble approach using different dropout patterns.
        Full ensemble would require multiple separately trained models.
        """
        ensemble_settings = self.ensemble_config
        # uncertainty.method 'ensemble' requires the ensemble section to be
        # enabled; a disabled or missing section is a configuration error,
        # never a deterministic prediction.
        if ensemble_settings.get('enabled') is not True:
            raise ValueError(
                "uncertainty.method 'ensemble' requires ensemble.enabled = true"
            )
        self._require_active_dropout('ensemble')
        n_members = self._require_positive_count(
            ensemble_settings, 'n_members', 'ensemble.n_members'
        )
        seed = ensemble_settings.get('seed')
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(
                f"ensemble.seed must be an explicit integer, got {seed!r}"
            )
        
        input_tensor = input_tensor.to(self.device)
        
        predictions = []
        
        # Member i draws its dropout masks from seed + i on a forked generator
        # state, so the members are reproducible and the caller's global
        # Torch RNG streams are left unchanged. Only the forked generator of
        # the model device is seeded: torch.manual_seed would also reseed
        # every CUDA generator outside the fork.
        fork_devices = [] if self.device.type == 'cpu' else [self.device]
        with torch.random.fork_rng(devices=fork_devices):
            with self._dropout_sampling_mode():
                with torch.no_grad():
                    for i in range(n_members):
                        if self.device.type == 'cuda':
                            with torch.cuda.device(self.device):
                                torch.cuda.manual_seed(seed + i)
                        else:
                            torch.default_generator.manual_seed(seed + i)
                        output = self.model(input_tensor)
                        predictions.append(output)
        
        # Stack and compute ensemble mean
        predictions = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(predictions, dim=0)
        
        # Store uncertainty information
        self._last_prediction_uncertainty = {
            'mean': mean_prediction,
            'std': torch.std(predictions, dim=0),
            'samples': predictions,
            'method': 'ensemble'
        }
        
        return mean_prediction
    
    @staticmethod
    def _require_positive_count(settings: Dict[str, Any], key: str, label: str) -> int:
        """Read an explicit positive sample count (no default count)."""
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be an explicit positive integer, got {value!r}")
        return value

    def get_prediction_uncertainty(self) -> Optional[Dict[str, Tensor]]:
        """Get uncertainty information from the last prediction.
        
        Returns:
            Optional[Dict[str, Tensor]]: Dictionary containing:
                - 'mean': Mean prediction
                - 'std': Standard deviation of predictions
                - 'samples': All prediction samples
                - 'method': Uncertainty method used
        """
        return getattr(self, '_last_prediction_uncertainty', None)
    
    def train(self,
              training_data: Union[SurrogateDataset, Tuple[List[StateDict], List[FluxDict]]],
              validation_data: Optional[Union[SurrogateDataset, Tuple[List[StateDict], List[FluxDict]]]] = None,
              epochs: Optional[int] = None,
              verbose: bool = True,
              validate_units: bool = True) -> Dict[str, Any]:
        """Train the surrogate model.
        
        Parameters:
            training_data : SurrogateDataset or Tuple[List[StateDict], List[FluxDict]]
                Training dataset or (states, fluxes) tuple
            validation_data : Optional, same format as training_data
                Validation dataset for early stopping
            epochs : int, optional
                Number of epochs (overrides config)
            verbose : bool
                Whether to print training progress
                
        Returns:
            Dict[str, Any]: Training results and history
        """
        if self.model is None:
            raise RuntimeError("Model not created")

        training_dataset = self._ensure_surrogate_dataset(training_data, validate_units=validate_units)
        validation_dataset: Optional[SurrogateDataset] = None
        if validation_data is not None:
            validation_dataset = self._ensure_surrogate_dataset(validation_data, validate_units=validate_units)

        batch_size = int(self._training_setting('batch_size'))
        # An explicit epochs argument overrides the configured (or YAML) value;
        # epochs=0 is rejected, not read as 'use the configured value'.
        if epochs is None:
            epochs = self._training_setting('epochs')
        if isinstance(epochs, bool) or not isinstance(epochs, (int, np.integer)) or epochs < 1:
            raise ValueError(f"epochs must be a positive integer, got {epochs!r}")
        epochs = int(epochs)
        early_stopping_patience = int(self._training_setting('early_stopping_patience'))

        train_loader, val_loader = self._build_train_val_loaders(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            batch_size=batch_size,
        )

        best_val_loss = float('inf')
        patience_counter = 0
        self.model.train()

        for epoch in range(epochs):
            avg_train_loss = self._run_training_epoch(train_loader)
            self.training_history['train_loss'].append(avg_train_loss)

            avg_val_loss: Optional[float] = None
            if val_loader is not None:
                avg_val_loss = self._run_validation_epoch(val_loader)
                self.training_history['val_loss'].append(avg_val_loss)
                best_val_loss, patience_counter, should_stop = self._update_early_stopping_state(
                    avg_val_loss=avg_val_loss,
                    best_val_loss=best_val_loss,
                    patience_counter=patience_counter,
                    early_stopping_patience=early_stopping_patience,
                    epoch=epoch,
                    verbose=verbose,
                )
                if should_stop:
                    break
                self.model.train()

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.training_history['learning_rate'].append(current_lr)

            self._log_training_progress(
                epoch=epoch,
                epochs=epochs,
                avg_train_loss=avg_train_loss,
                avg_val_loss=avg_val_loss,
                verbose=verbose,
            )
            self.epochs_trained += 1

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        self.is_trained = True
        self.model.eval()
        self.save_model()

        return {
            'epochs_trained': self.epochs_trained,
            'final_train_loss': self.training_history['train_loss'][-1] if self.training_history['train_loss'] else None,
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
            'best_val_loss': best_val_loss if val_loader is not None else None,
            'training_history': self.training_history
        }

    def _ensure_surrogate_dataset(
        self,
        data: Union[SurrogateDataset, Tuple[List[StateDict], List[FluxDict]]],
        *,
        validate_units: bool,
    ) -> SurrogateDataset:
        if isinstance(data, SurrogateDataset):
            return data
        states, fluxes = data
        return SurrogateDataset(
            states,
            fluxes,
            self.input_variables,
            self.output_variables,
            input_units=self._get_expected_input_units(),
            output_units=self._get_expected_output_units(),
            unit_conversions=self.units_conversion,
            validate_units=validate_units,
        )

    def _build_train_val_loaders(
        self,
        *,
        training_dataset: SurrogateDataset,
        validation_dataset: Optional[SurrogateDataset],
        batch_size: int,
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        training_settings = self.training_config
        shuffle_training = self._training_setting("shuffle_training")
        if not isinstance(shuffle_training, bool):
            raise TypeError(f"training.shuffle_training must be boolean, got {shuffle_training!r}")
        raw_num_workers = self._training_setting("num_workers")
        if (
            isinstance(raw_num_workers, bool)
            or not isinstance(raw_num_workers, int)
            or raw_num_workers < 0
        ):
            raise ValueError("training.num_workers must be a nonnegative integer")
        num_workers = int(raw_num_workers)

        pin_memory = self._training_setting("pin_memory")
        persistent_workers = self._training_setting("persistent_workers")
        if not isinstance(pin_memory, bool):
            raise TypeError("training.pin_memory must be boolean")
        if not isinstance(persistent_workers, bool):
            raise TypeError("training.persistent_workers must be boolean")

        raw_prefetch = self._training_setting("prefetch_factor")
        if (
            isinstance(raw_prefetch, bool)
            or not isinstance(raw_prefetch, int)
            or raw_prefetch <= 0
        ):
            raise ValueError("training.prefetch_factor must be a positive integer")
        prefetch_factor = int(raw_prefetch) if num_workers > 0 else None
        effective_persistent = persistent_workers and num_workers > 0

        raw_seed = self._training_setting("seed")
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise ValueError("training.seed must be an integer")
        seed = int(raw_seed)

        # A declared validation_split holds out that fraction of the training
        # samples, drawn with a generator seeded by training.seed, when no
        # validation dataset is given; explicit validation data takes precedence.
        raw_split = training_settings.get("validation_split")
        if raw_split is not None:
            if (
                isinstance(raw_split, bool)
                or not isinstance(raw_split, (int, float))
                or not 0.0 < float(raw_split) < 1.0
            ):
                raise ValueError(
                    f"training.validation_split must be a number in (0, 1), got {raw_split!r}"
                )
            if validation_dataset is None:
                n_total = len(training_dataset)
                n_validation = int(round(float(raw_split) * n_total))
                if not 1 <= n_validation < n_total:
                    raise ValueError(
                        f"training.validation_split={raw_split} leaves no training or no "
                        f"validation sample out of {n_total}"
                    )
                split_generator = torch.Generator(device="cpu")
                split_generator.manual_seed(seed)
                training_dataset, validation_dataset = random_split(
                    training_dataset,
                    [n_total - n_validation, n_validation],
                    generator=split_generator,
                )

        def loader_kwargs(*, generator_seed: int) -> Dict[str, Any]:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(generator_seed)
            kwargs: Dict[str, Any] = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
                "generator": generator,
            }
            if num_workers > 0:
                kwargs["persistent_workers"] = effective_persistent
                kwargs["prefetch_factor"] = prefetch_factor
                kwargs["worker_init_fn"] = _initialize_dataloader_worker
            return kwargs

        train_loader = DataLoader(
            training_dataset,
            shuffle=shuffle_training,
            **loader_kwargs(generator_seed=seed),
        )
        val_loader: Optional[DataLoader] = None
        if validation_dataset is not None:
            val_loader = DataLoader(
                validation_dataset,
                shuffle=False,
                **loader_kwargs(generator_seed=seed + 1),
            )
        self._dataloader_execution_info = {
            "requested_num_workers": int(raw_num_workers),
            "effective_num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": effective_persistent,
            "prefetch_factor": prefetch_factor,
            "seed": seed,
        }
        return train_loader, val_loader

    def _move_batch_to_device_dtype(self, batch_inputs: Tensor, batch_targets: Tensor) -> Tuple[Tensor, Tensor]:
        if self.model is None:
            raise RuntimeError("Model not created")
        batch_inputs = batch_inputs.to(self.device)
        batch_targets = batch_targets.to(self.device)
        target_dtype = next(self.model.parameters()).dtype
        if batch_inputs.dtype != target_dtype:
            batch_inputs = batch_inputs.to(dtype=target_dtype)
        if batch_targets.dtype != target_dtype:
            batch_targets = batch_targets.to(dtype=target_dtype)
        return batch_inputs, batch_targets

    def _compute_training_loss(self, outputs: Tensor, batch_targets: Tensor) -> Tensor:
        loss = self.loss_fn(outputs, batch_targets)
        if self.physics_loss_config:
            loss = loss + self._compute_physics_loss(outputs, batch_targets)
        return loss

    def _run_training_epoch(self, train_loader: DataLoader) -> float:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Model or optimizer not initialized")
        train_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = self._move_batch_to_device_dtype(batch_inputs, batch_targets)
            self.optimizer.zero_grad()
            outputs = self.model(batch_inputs)
            loss = self._compute_training_loss(outputs, batch_targets)
            loss.backward()
            self.optimizer.step()
            train_loss += float(loss.item())
        return train_loss / len(train_loader)

    def _run_validation_epoch(self, val_loader: DataLoader) -> float:
        if self.model is None:
            raise RuntimeError("Model not created")
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs, batch_targets = self._move_batch_to_device_dtype(batch_inputs, batch_targets)
                outputs = self.model(batch_inputs)
                loss = self.loss_fn(outputs, batch_targets)
                val_loss += float(loss.item())
        return val_loss / len(val_loader)

    def _update_early_stopping_state(
        self,
        *,
        avg_val_loss: float,
        best_val_loss: float,
        patience_counter: int,
        early_stopping_patience: int,
        epoch: int,
        verbose: bool,
    ) -> Tuple[float, int, bool]:
        if self.model is None:
            raise RuntimeError("Model not created")
        if avg_val_loss < best_val_loss:
            # Snapshot: state_dict() tensors share storage with the live
            # parameters, so a shallow dict copy would follow later updates.
            self.best_model_state = {
                key: value.detach().clone() for key, value in self.model.state_dict().items()
            }
            return avg_val_loss, 0, False

        updated_patience = patience_counter + 1
        if updated_patience >= early_stopping_patience:
            if verbose:
                logger.info(f"Early stopping at epoch {epoch+1}")
            return best_val_loss, updated_patience, True
        return best_val_loss, updated_patience, False

    @staticmethod
    def _log_training_progress(
        *,
        epoch: int,
        epochs: int,
        avg_train_loss: float,
        avg_val_loss: Optional[float],
        verbose: bool,
    ) -> None:
        if not verbose or (epoch + 1) % 10 != 0:
            return
        msg = f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.6f}"
        if avg_val_loss is not None:
            msg += f", Val loss: {avg_val_loss:.6f}"
        logger.info(msg)
    
    def _compute_physics_loss(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """Compute enhanced physics-informed loss terms.
        
        Implements various physical constraints and scientific principles
        to ensure surrogate model predictions are physically consistent.
        
        Parameters:
            outputs : Tensor
                Model predictions, shape [batch_size, output_dim]
            targets : Tensor
                Target values, shape [batch_size, output_dim]
                
        Returns:
            Tensor: Combined physics loss term
            
        References:
            - Raissi, M. et al. (2019). Physics-informed neural networks
            - Karniadakis, G.E. et al. (2021). Physics-informed ML
            - Wang, S. et al. (2021). Understanding and mitigating gradient flow pathologies
        """
        physics_settings = self._get_physics_settings()
        physics_loss = self._zero_scalar_like(outputs)
        loss_components: Dict[str, float] = {}

        component_losses = {
            "conservation": self._conservation_loss(outputs, targets, physics_settings),
            "smoothness": self._smoothness_loss(outputs, targets, physics_settings),
            "monotonicity": self._monotonicity_loss(outputs, physics_settings),
            "bounds": self._bounds_loss(outputs, physics_settings),
            "thermodynamic": self._thermodynamic_loss(outputs, physics_settings),
        }
        for name, component in component_losses.items():
            if component is None:
                continue
            physics_loss = physics_loss + component
            loss_components[name] = float(component.detach().item())

        existing_components = getattr(self, "_physics_loss_components", {})
        if isinstance(existing_components, dict):
            existing_components.update(loss_components)
            self._physics_loss_components = existing_components
        else:
            self._physics_loss_components = loss_components

        return self._apply_adaptive_physics_scaling(physics_loss, targets, physics_settings)

    @staticmethod
    def _zero_scalar_like(reference: Tensor) -> Tensor:
        return torch.zeros((), device=reference.device, dtype=reference.dtype)

    def _get_physics_settings(self) -> Dict[str, Any]:
        return self.physics_loss_config

    def _conservation_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        physics_settings: Dict[str, Any],
    ) -> Optional[Tensor]:
        if "conservation_weight" not in physics_settings:
            return None
        conservation_type = str(
            physics_settings.get("conservation_type", "declared_ledger")
        )
        if conservation_type != "declared_ledger":
            raise ValueError(
                "conservation_type must be 'declared_ledger'; inferred sum, "
                "mass_balance, and energy_balance laws are not supported"
            )
        if self._conservation_matrix is None:
            raise ValueError(
                "conservation_weight requires an explicit conservation_matrix"
            )
        if outputs.ndim != 2 or outputs.numel() == 0:
            raise ValueError("conservation outputs must be a non-empty rank-2 tensor")

        weight = torch.as_tensor(
            physics_settings.get("conservation_weight"),
            device=outputs.device,
            dtype=outputs.dtype,
        )
        if weight.numel() != 1 or not bool(torch.isfinite(weight).all().item()):
            raise ValueError("conservation_weight must be a finite non-negative scalar")
        if bool((weight < 0).item()):
            raise ValueError("conservation_weight must be a finite non-negative scalar")

        ledger = self._conservation_matrix.to(
            device=outputs.device,
            dtype=outputs.dtype,
        )
        if ledger.shape[1] != outputs.shape[1]:
            raise ValueError(
                "conservation_matrix column count must match runtime outputs"
            )

        # References: Karniadakis et al. (2021), Nat. Rev. Phys. 3, 422-440.
        # Mathematical expression: L_C = lambda_C mean((Y C^T)^2).
        # Parameters: Y is the output matrix, C is the declared conservation
        # ledger, and lambda_C is the non-negative penalty weight.
        residual = outputs @ ledger.transpose(0, 1)
        return weight * residual.pow(2).mean()

    def _smoothness_loss(
        self,
        outputs: Tensor,
        targets: Tensor,
        physics_settings: Dict[str, Any],
    ) -> Optional[Tensor]:
        if "smoothness_weight" not in physics_settings:
            return None
        weight = float(physics_settings["smoothness_weight"])
        smoothness_type = str(physics_settings.get("smoothness_type", "temporal"))
        if smoothness_type not in ("temporal", "spatial", "adaptive"):
            raise ValueError(
                "smoothness_type must be 'temporal', 'spatial' or 'adaptive', "
                f"got {smoothness_type!r}"
            )
        batch_size, output_dim = outputs.shape
        if smoothness_type == "temporal" and batch_size > 1:
            gradients = outputs[1:] - outputs[:-1]
            return weight * gradients.pow(2).mean()
        if smoothness_type == "spatial" and output_dim > 1:
            gradients = outputs[:, 1:] - outputs[:, :-1]
            return weight * gradients.pow(2).mean()
        if smoothness_type == "adaptive" and batch_size > 1:
            target_variance = torch.var(targets, dim=0, keepdim=True)
            gradients = outputs[1:] - outputs[:-1]
            normalized_weight = weight / (target_variance + 1e-8)
            return (normalized_weight * gradients.pow(2)).mean()
        # One sample (temporal/adaptive) or one output (spatial): no pair of
        # neighbours exists, so the smoothness penalty is zero.
        return self._zero_scalar_like(outputs)

    def _monotonicity_loss(self, outputs: Tensor, physics_settings: Dict[str, Any]) -> Optional[Tensor]:
        if "monotonicity_weight" not in physics_settings:
            return None
        batch_size, output_dim = outputs.shape
        constraints = self._require_constraints(
            physics_settings, "monotonicity_constraints", output_dim
        )
        for constraint in constraints:
            direction = constraint.get("direction")
            if direction not in ("increasing", "decreasing"):
                raise ValueError(
                    "monotonicity constraint direction must be 'increasing' or "
                    f"'decreasing', got {direction!r}"
                )

        weight = float(physics_settings["monotonicity_weight"])
        if batch_size <= 1:
            # A single sample has no ordered pair to penalize.
            return self._zero_scalar_like(outputs)

        monotonicity_sum = self._zero_scalar_like(outputs)
        for constraint in constraints:
            output_idx = constraint["output_idx"]
            if constraint["direction"] == "increasing":
                violations = torch.relu(outputs[:-1, output_idx] - outputs[1:, output_idx])
            else:
                violations = torch.relu(outputs[1:, output_idx] - outputs[:-1, output_idx])
            monotonicity_sum = monotonicity_sum + violations.sum()

        return weight * monotonicity_sum / len(constraints)

    def _bounds_loss(self, outputs: Tensor, physics_settings: Dict[str, Any]) -> Optional[Tensor]:
        if "bounds_weight" not in physics_settings:
            return None
        output_dim = outputs.shape[1]
        constraints = self._require_constraints(
            physics_settings, "bounds_constraints", output_dim
        )
        for constraint in constraints:
            if constraint.get("lower_bound") is None and constraint.get("upper_bound") is None:
                raise ValueError(
                    "each bounds constraint needs a lower_bound or an upper_bound"
                )

        weight = float(physics_settings["bounds_weight"])
        bounds_sum = self._zero_scalar_like(outputs)
        for constraint in constraints:
            output_idx = constraint["output_idx"]
            lower_bound = constraint.get("lower_bound")
            upper_bound = constraint.get("upper_bound")
            if lower_bound is not None:
                lower_tensor = torch.as_tensor(lower_bound, device=outputs.device, dtype=outputs.dtype)
                bounds_sum = bounds_sum + torch.relu(lower_tensor - outputs[:, output_idx]).sum()
            if upper_bound is not None:
                upper_tensor = torch.as_tensor(upper_bound, device=outputs.device, dtype=outputs.dtype)
                bounds_sum = bounds_sum + torch.relu(outputs[:, output_idx] - upper_tensor).sum()

        return weight * bounds_sum / len(constraints)

    @staticmethod
    def _require_constraints(
        physics_settings: Dict[str, Any], key: str, output_dim: int
    ) -> List[Dict[str, Any]]:
        """Validate a declared constraint list; nothing is skipped silently.

        A weighted term needs a non-empty list of mappings, each with an
        explicit integer output_idx inside the output dimension.
        """
        constraints = physics_settings.get(key)
        if not isinstance(constraints, list) or not constraints:
            raise ValueError(
                f"a weighted physics term needs a non-empty {key} list"
            )
        for constraint in constraints:
            if not isinstance(constraint, dict):
                raise ValueError(
                    f"each entry of {key} must be a mapping, got {constraint!r}"
                )
            output_idx = constraint.get("output_idx")
            if (
                isinstance(output_idx, bool)
                or not isinstance(output_idx, int)
                or not 0 <= output_idx < output_dim
            ):
                raise ValueError(
                    f"{key} entry needs an explicit output_idx in [0, {output_dim}), "
                    f"got {output_idx!r}"
                )
        return constraints

    def _thermodynamic_loss(self, outputs: Tensor, physics_settings: Dict[str, Any]) -> Optional[Tensor]:
        if "thermodynamic_weight" not in physics_settings:
            return None
        # No thermodynamic response law is implemented; a declared weight is
        # an error instead of a term that is always zero.
        raise NotImplementedError(
            "thermodynamic_weight is declared but no thermodynamic physics-loss "
            "term is implemented"
        )

    def _apply_adaptive_physics_scaling(
        self,
        physics_loss: Tensor,
        targets: Tensor,
        physics_settings: Dict[str, Any],
    ) -> Tensor:
        # Rescaling the declared weights is opt-in: without an explicit
        # adaptive_scaling = true the declared weights are used as given.
        adaptive_scaling = physics_settings.get("adaptive_scaling", False)
        if not isinstance(adaptive_scaling, bool):
            raise TypeError("physics_loss.adaptive_scaling must be boolean")
        if not adaptive_scaling:
            return physics_loss
        if float(physics_loss.detach().item()) <= 0.0:
            return physics_loss

        data_loss_scale = torch.abs(targets).mean()
        physics_loss_scale = torch.abs(physics_loss)
        if float(physics_loss_scale.detach().item()) <= float((data_loss_scale * 10.0).detach().item()):
            return physics_loss
        scaling_factor = (data_loss_scale * 2.0) / (physics_loss_scale + 1e-8)
        return physics_loss * scaling_factor
    
    def get_physics_loss_components(self) -> Dict[str, float]:
        """Get breakdown of physics loss components for monitoring.
        
        Returns:
            Dict[str, float]: Physics loss components by type
        """
        return getattr(self, '_physics_loss_components', {})
    
    def save_model(self, path: Optional[Path] = None) -> None:
        """Save model checkpoint.
        
        Parameters:
            path : Path, optional
                Save path (defaults to self.model_path)
        """
        if self.model is None:
            raise RuntimeError("No model to save")
        
        save_path = path or self.model_path
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'architecture_config': self.architecture_config,
            'training_config': self.training_config,
            'input_variables': self.input_variables,
            'output_variables': self.output_variables,
            'epochs_trained': self.epochs_trained,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        # Add optimizer state if available
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def get_gradients(self, input_tensor: Tensor) -> Dict[str, Tensor]:
        """Compute gradients of outputs with respect to inputs.
        
        Parameters:
            input_tensor : Tensor
                Input features requiring gradients
                
        Returns:
            Dict[str, Tensor]: Gradients for each output variable
        """
        if self.model is None:
            raise RuntimeError("Model not created")
        
        # Ensure input requires gradients
        target_dtype = next(self.model.parameters()).dtype
        input_tensor = input_tensor.to(device=self.device, dtype=target_dtype)
        input_tensor.requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        outputs = self.model(input_tensor)
        
        # Compute gradients for each output
        gradients = {}
        
        for i, var_name in enumerate(self.output_variables):
            # Select output column
            output_i = outputs[:, i].sum()  # Sum for batch
            
            # Compute gradient
            grad = torch.autograd.grad(
                output_i,
                input_tensor,
                retain_graph=True,
                create_graph=False
            )[0]
            
            gradients[var_name] = grad
        
        return gradients
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        base_info = super().get_model_info()
        
        pytorch_info = {
            'is_trained': self.is_trained,
            'epochs_trained': self.epochs_trained,
            'architecture': self.architecture_config,
            'training_config': self.training_config,
            'model_parameters': self.count_parameters() if self.model else 0,
            'has_optimizer': self.optimizer is not None,
            'has_scheduler': self.scheduler is not None
        }
        
        return {**base_info, **pytorch_info}
    
    def count_parameters(self) -> int:
        """Count trainable parameters in model."""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    @staticmethod
    def _coerce_unit_mapping(
        payload: Any,
        label: str,
    ) -> Dict[str, str]:
        """Normalize one explicit unit mapping without inventing defaults."""
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise TypeError(f"{label} must be a mapping")
        normalized: Dict[str, str] = {}
        for variable, unit in payload.items():
            if not isinstance(variable, str) or not variable.strip():
                raise TypeError(f"{label} keys must be non-empty strings")
            if not isinstance(unit, str) or not unit.strip():
                raise TypeError(
                    f"{label} value for {variable} must be a non-empty string"
                )
            normalized[variable] = unit
        return normalized

    def _validate_unit_configuration(self) -> None:
        """Validate explicit declarations against SCM and conversion direction."""
        input_names = set(self.input_variables)
        output_names = set(self.output_variables)
        extra_inputs = sorted(set(self.input_units_config) - input_names)
        extra_outputs = sorted(set(self.output_units_config) - output_names)
        if extra_inputs:
            raise ValueError(
                f"input_units contains undeclared variables: {extra_inputs}"
            )
        if extra_outputs:
            raise ValueError(
                f"output_units contains undeclared variables: {extra_outputs}"
            )

        for role, configured_units in (
            ("input", self.input_units_config),
            ("output", self.output_units_config),
        ):
            for variable, unit in configured_units.items():
                SurrogateDataset._require_valid_unit(role, variable, unit)
                contract_unit = UnitSystem.get_contract_unit(variable)
                if contract_unit is not None and unit != contract_unit:
                    raise ValueError(
                        f"Explicit {role} unit for {variable} ({unit}) "
                        f"conflicts with SCM unit ({contract_unit})"
                    )

        declared_names = input_names | output_names
        extra_conversions = sorted(set(self.units_conversion) - declared_names)
        if extra_conversions:
            raise ValueError(
                "units_conversion contains undeclared variables: "
                f"{extra_conversions}"
            )
        for variable, conversion in self.units_conversion.items():
            if (
                not isinstance(conversion, (tuple, list))
                or len(conversion) != 2
            ):
                raise ValueError(
                    f"Unit conversion for {variable} must be a two-item pair"
                )
            source_unit, target_unit = conversion
            SurrogateDataset._require_valid_unit(
                "conversion source",
                variable,
                source_unit,
            )
            SurrogateDataset._require_valid_unit(
                "conversion target",
                variable,
                target_unit,
            )
            if variable in input_names:
                external_input_unit = (
                    self.input_units_config.get(variable)
                    or UnitSystem.get_contract_unit(variable)
                )
                if (
                    external_input_unit is not None
                    and source_unit != external_input_unit
                ):
                    raise ValueError(
                        f"Input conversion for {variable} must start from "
                        f"declared unit {external_input_unit}"
                    )
            if variable in output_names:
                external_output_unit = (
                    self.output_units_config.get(variable)
                    or UnitSystem.get_contract_unit(variable)
                )
                if (
                    external_output_unit is not None
                    and target_unit != external_output_unit
                ):
                    raise ValueError(
                        f"Output conversion for {variable} must end in "
                        f"declared unit {external_output_unit}"
                    )

    def _resolve_variable_units(
        self,
        variables: List[str],
        configured_units: Dict[str, str],
        role: str,
    ) -> Dict[str, str]:
        """Resolve each unit from explicit configuration or an exact SCM."""
        resolved: Dict[str, str] = {}
        for variable in variables:
            unit = configured_units.get(variable)
            if unit is None:
                unit = UnitSystem.get_contract_unit(variable)
            if unit is None:
                raise ValueError(
                    f"No unit declaration or SCM contract for {role} "
                    f"variable '{variable}'"
                )
            SurrogateDataset._require_valid_unit(role, variable, unit)
            resolved[variable] = unit
        return resolved

    def _get_expected_input_units(self) -> Dict[str, str]:
        """Return declared external units for all model inputs."""
        return self._resolve_variable_units(
            self.input_variables,
            self.input_units_config,
            "input",
        )

    def _get_expected_output_units(self) -> Dict[str, str]:
        """Return declared external units for all model outputs."""
        return self._resolve_variable_units(
            self.output_variables,
            self.output_units_config,
            "output",
        )

# Export public API
__all__ = [
    'PyTorchSurrogate',
    'FlexibleSurrogateNet',
    'SurrogateDataset'
]
