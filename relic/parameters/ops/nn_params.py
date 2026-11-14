from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List, Mapping, Sequence, Union, cast

import torch
import torch.nn as nn

from ..spec import ParameterSpec

logger = logging.getLogger(__name__)


def register_nn_parameter(manager: Any, spec: ParameterSpec) -> None:
    """Register parameter as PyTorch nn.Parameter."""
    dtype_str = (spec.dtype or "").lower()
    if isinstance(spec.value, str) or dtype_str in {"string", "str", "category", "enum", "object"}:
        raise TypeError(
            f"Cannot register non-numeric parameter {spec.key} (dtype '{spec.dtype}') "
            "as a learnable nn.Parameter."
        )
    param_tensor = spec.to_tensor(manager.device)
    param_obj = nn.Parameter(param_tensor)
    manager.nn_parameters[spec.key] = param_obj
    logger.debug("Registered nn.Parameter: %s", spec.key)


def bind_learnable_value(manager: Any, spec: ParameterSpec) -> None:
    """Keep a parameter's live binding in step with its specification value.

    A learnable specification gets a binding that holds its value: an existing
    binding of the same shape and dtype keeps its Parameter object and gradient
    flag and takes the value in place, otherwise a new binding is registered.
    A specification that is not learnable has no binding. A learnable
    specification without a tensor form raises TypeError.
    """
    if not spec.requires_grad:
        manager.nn_parameters.pop(spec.key, None)
        return
    value = spec.to_tensor(manager.device)
    binding = manager.nn_parameters.get(spec.key)
    if binding is not None and binding.shape == value.shape and binding.dtype == value.dtype:
        with torch.no_grad():
            binding.copy_(value)
    else:
        register_nn_parameter(manager, spec)


def capture_binding_values(manager: Any, keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    """Detached copies of the live values of the named learnable bindings.

    An optimizer may have moved a binding away from its declared (spec)
    value; the capture keeps the live value, as the manager snapshot does for
    every binding. A key without a binding raises KeyError.
    """
    return {key: manager.nn_parameters[key].detach().clone() for key in keys}


def restore_binding_values(manager: Any, values: Mapping[str, torch.Tensor]) -> None:
    """Write live values back into the named learnable bindings in place.

    Each binding keeps its object identity, gradient flag and device; a value
    is copied across devices. Every value is checked before any binding
    changes: a key without a binding raises KeyError, a value that is no
    tensor of the binding's shape and dtype raises ValueError.
    """
    bindings = {key: manager.nn_parameters[key] for key in values}
    for key, value in values.items():
        binding = bindings[key]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != binding.shape
            or value.dtype != binding.dtype
        ):
            raise ValueError(
                f"Live value for learnable binding {key} does not match its shape and dtype"
            )
    with torch.no_grad():
        for key, value in values.items():
            bindings[key].copy_(value)


def relocate_binding(manager: Any, key: str, device: Union[str, torch.device]) -> bool:
    """Move one learnable binding to a device in place, keeping its identity.

    Every holder of the binding reads the moved tensor and its gradient moves
    with it. Returns True when the binding moved and False when it already
    was on the device. A key without a binding raises KeyError.
    """
    binding = manager.nn_parameters[key]
    target = torch.empty(0, device=torch.device(device)).device
    if binding.device == target:
        return False
    with torch.no_grad():
        binding.data = binding.data.to(device=target)
        if binding.grad is not None:
            binding.grad = binding.grad.to(device=target)
    return True


def get_nn_parameters(manager: Any) -> Dict[str, nn.Parameter]:
    """Get all PyTorch parameters for optimization."""
    nn_params = manager.nn_parameters.copy()
    if isinstance(nn_params, dict):
        return cast(Dict[str, nn.Parameter], nn_params)
    return {}


def get_trainable_parameters(manager: Any) -> Dict[str, nn.Parameter]:
    """
    Get only trainable PyTorch parameters (requires_grad=True).

    This method returns parameters that are marked as learnable in their
    ParameterSpec and are ready for gradient-based optimization.

    Mathematical Expression
    -----------------------
        Let Theta be the set of all parameters, and L subset Theta be learnable parameters.
        This method returns: {theta_i in L | spec(theta_i).requires_grad = True}

    Returns
    -------
    Dict[str, nn.Parameter]
        Dictionary mapping parameter keys to trainable nn.Parameter objects

    References
    ----------
    - Kingma and Ba (2015). Adam: A Method for Stochastic Optimization
    - Goodfellow et al. (2016). Deep Learning, Chapter 8: Optimization

    Notes
    -----
    Only parameters with ParameterSpec.requires_grad=True and successfully
    registered as nn.Parameter are included. Use this method when setting
    up optimizers for parameter calibration or data assimilation.
    """
    trainable_params: Dict[str, nn.Parameter] = {}

    if not getattr(manager, "_preloaded", False) and getattr(manager, "_preload_all_default", False):
        manager._preload_parameters(warmup_cache=False)

    for key, spec in manager.parameters.items():
        if spec.requires_grad and key not in manager.nn_parameters:
            try:
                register_nn_parameter(manager, spec)
            except TypeError as exc:
                logger.warning(
                    "Skipping non-numeric learnable parameter %s: %s",
                    spec.key,
                    exc,
                )

    for key, spec in manager.parameters.items():
        if spec.requires_grad and key in manager.nn_parameters:
            param = manager.nn_parameters[key]
            if param.requires_grad:
                trainable_params[key] = param

    logger.debug("Retrieved %d trainable parameters", len(trainable_params))
    return trainable_params


def freeze(manager: Any, keys: Optional[List[str]] = None) -> None:
    """
    Freeze parameters (disable gradients).

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    keys : Optional[List[str]]
        Parameter keys to freeze, all learnable if None.
    """
    target_keys = keys or [k for k, spec in manager.parameters.items() if spec.requires_grad]

    for key in target_keys:
        if key in manager.nn_parameters:
            manager.nn_parameters[key].requires_grad_(False)
            logger.debug("Froze parameter: %s", key)


def unfreeze(manager: Any, keys: Optional[List[str]] = None) -> None:
    """
    Unfreeze parameters (enable gradients).

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    keys : Optional[List[str]]
        Parameter keys to unfreeze, all learnable if None.
    """
    target_keys = keys or [k for k, spec in manager.parameters.items() if spec.requires_grad]

    for key in target_keys:
        if key in manager.nn_parameters:
            manager.nn_parameters[key].requires_grad_(True)
            logger.debug("Unfroze parameter: %s", key)


def to_device(
    manager: Any,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
) -> Any:
    """
    Move all PyTorch parameters to specified device/dtype.

    Parameters
    ----------
    manager : Any
        ParameterManager instance.
    device : Union[str, torch.device]
        Target device.
    dtype : Optional[torch.dtype]
        Target dtype.

    Returns
    -------
    Any
        Manager reference for chaining.
    """
    if isinstance(device, str):
        device = torch.device(device)

    manager.device = device

    for key, param in manager.nn_parameters.items():
        if dtype is not None:
            param.data = param.data.to(device=device, dtype=dtype)
        else:
            param.data = param.data.to(device=device)

    logger.info("Moved parameters to device: %s", device)
    return manager
