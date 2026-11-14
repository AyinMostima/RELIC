"""
Tensor context management for consistent device and dtype handling.

Provides unified management of tensor device, dtype, and gradient
requirements across all coupling operations.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, TYPE_CHECKING
import torch
from relic.core import device_policy
from torch import Tensor

if TYPE_CHECKING:
    from relic.coupling.core.policy import ScalingMode, PrecisionPolicy  # pragma: no cover

logger = logging.getLogger(__name__)


@dataclass
class TensorContext:
    """
    Context for tensor operations.
    
    Attributes:
        device : torch.device
            Device for tensor operations
        dtype : torch.dtype
            Data type for tensors
        requires_grad : bool
            Whether tensors require gradients
        memory_format : torch.memory_format
            Memory format for tensors
    """
    device: torch.device
    dtype: torch.dtype
    requires_grad: bool = False
    memory_format: torch.memory_format = torch.contiguous_format
    
    def apply(self, tensor: Tensor) -> Tensor:
        """
        Apply context to a tensor.
        
        Parameters:
            tensor : Tensor
                Input tensor
                
        Returns:
            Tensor: Tensor with context applied
        """
        # Move to device if needed
        if tensor.device != self.device:
            tensor = tensor.to(device=self.device)
        
        # Convert dtype if needed
        if tensor.dtype != self.dtype:
            tensor = tensor.to(dtype=self.dtype)
        
        # Set gradient requirement
        if tensor.requires_grad != self.requires_grad:
            tensor = tensor.requires_grad_(self.requires_grad)
        
        # Ensure memory format
        if not tensor.is_contiguous(memory_format=self.memory_format):
            tensor = tensor.contiguous(memory_format=self.memory_format)
        
        return tensor
    
    def apply_dict(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply context to a dictionary of tensors.
        
        Parameters:
            data : Dict[str, Tensor]
                Dictionary of tensors
                
        Returns:
            Dict[str, Tensor]: Dictionary with context applied
        """
        return {key: self.apply(tensor) for key, tensor in data.items()}
    
    def create_tensor(
        self,
        data: Union[List, Tensor],
        **kwargs
    ) -> Tensor:
        """
        Create a tensor with this context.
        
        Parameters:
            data : Union[List, Tensor]
                Data for tensor
            **kwargs : Any
                Additional tensor creation arguments
                
        Returns:
            Tensor: Created tensor with context
        """
        if isinstance(data, Tensor):
            return self.apply(data)
        
        # Create tensor with context
        tensor = torch.tensor(
            data,
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            **kwargs
        )
        
        return tensor.contiguous(memory_format=self.memory_format)
    
    def create_zeros(self, *size, **kwargs) -> Tensor:
        """Create zeros tensor with context."""
        return torch.zeros(
            *size,
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            **kwargs
        ).contiguous(memory_format=self.memory_format)
    
    def create_ones(self, *size, **kwargs) -> Tensor:
        """Create ones tensor with context."""
        return torch.ones(
            *size,
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            **kwargs
        ).contiguous(memory_format=self.memory_format)
    
    def create_empty(self, *size, **kwargs) -> Tensor:
        """Create empty tensor with context."""
        return torch.empty(
            *size,
            device=self.device,
            dtype=self.dtype,
            requires_grad=self.requires_grad,
            **kwargs
        ).contiguous(memory_format=self.memory_format)


def get_tensor_context(
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    requires_grad: bool = False,
    memory_format: torch.memory_format = torch.contiguous_format,
    mode: Optional["ScalingMode"] = None,  # type: ignore[FWDREF]
    precision_policy: Optional["PrecisionPolicy"] = None,  # type: ignore[FWDREF]
) -> TensorContext:
    """
    Get tensor context with defaults.
    
    Parameters:
        device : Optional[torch.device]
            Device (defaults to CUDA-first when available)
        dtype : Optional[torch.dtype]
            Data type (defaults to float32)
        requires_grad : bool
            Whether gradients are required
        memory_format : torch.memory_format
            Memory format
            
    Returns:
        TensorContext: The tensor context
    """
    if precision_policy is None and mode is not None:
        from relic.coupling.core.policy import get_precision_policy
        precision_policy = get_precision_policy(mode)

    if device is None:
        device = device_policy.resolve_device(default="auto")

    if dtype is None:
        dtype = precision_policy.dtype if precision_policy is not None else torch.float32
    
    return TensorContext(
        device=device,
        dtype=dtype,
        requires_grad=requires_grad,
        memory_format=memory_format
    )


def apply_context(
    data: Union[Tensor, Dict[str, Tensor]],
    context: TensorContext
) -> Union[Tensor, Dict[str, Tensor]]:
    """
    Apply context to data.
    
    Parameters:
        data : Union[Tensor, Dict[str, Tensor]]
            Input data
        context : TensorContext
            Context to apply
            
    Returns:
        Union[Tensor, Dict[str, Tensor]]: Data with context applied
    """
    if isinstance(data, dict):
        return context.apply_dict(data)
    else:
        return context.apply(data)


def infer_context(
    data: Union[Tensor, Dict[str, Tensor], List[Tensor]]
) -> TensorContext:
    """
    Infer context from existing data.
    
    Parameters:
        data : Union[Tensor, Dict[str, Tensor], List[Tensor]]
            Data to infer context from
            
    Returns:
        TensorContext: Inferred context
    """
    # Get first tensor
    if isinstance(data, Tensor):
        first_tensor = data
    elif isinstance(data, dict):
        first_tensor = next(iter(data.values()))
    elif isinstance(data, list):
        first_tensor = data[0]
    else:
        raise ValueError(f"Cannot infer context from type: {type(data)}")
    
    return TensorContext(
        device=first_tensor.device,
        dtype=first_tensor.dtype,
        requires_grad=first_tensor.requires_grad
    )


def ensure_same_context(*tensors: Tensor) -> bool:
    """
    Check if all tensors have the same context.
    
    Parameters:
        *tensors : Tensor
            Tensors to check
            
    Returns:
        bool: True if all have same context
    """
    if not tensors:
        return True
    
    first = tensors[0]
    first_device = first.device
    first_dtype = first.dtype
    
    for tensor in tensors[1:]:
        if tensor.device != first_device or tensor.dtype != first_dtype:
            return False
    
    return True


def align_contexts(*tensors: Tensor) -> List[Tensor]:
    """
    Align all tensors to the same context.
    
    Uses the context of the first tensor.
    
    Parameters:
        *tensors : Tensor
            Tensors to align
            
    Returns:
        List[Tensor]: Aligned tensors
    """
    if not tensors:
        return []
    
    context = infer_context(tensors[0])
    return [context.apply(tensor) for tensor in tensors]
