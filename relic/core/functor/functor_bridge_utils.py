"""Shared helpers for functor bridge classes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
from torch import Tensor

from .functor_base import _get_parameter_manager
from ...utils.tensor_ops import to_tensor as convert_to_tensor

logger = logging.getLogger(__name__)



def _ensure_tensor(
    value: Union[float, Tensor],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Convert numerical values to PyTorch tensors with device consistency.
    """
    return convert_to_tensor(value, dtype=dtype, device=device)


class BridgeParameterMixin:
    """Shared parameter access and conservation ledger helpers for bridge functors."""

    _conservation_targets: Dict[str, List[str]] = {}

    def normalize_parameter_lookup_kwargs(
        self,
        kwargs: Dict[str, Any],
        *,
        namespace: Optional[str] = None,
        source_order: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, Any]:
        """
        Supply the class namespace and source order to get_parameter_value kwargs.

        An explicit ``namespace`` or ``source_order`` of the caller wins; every
        other option (units, bounds, validate, required, expected_type) is
        forwarded unchanged, and an option get_parameter_value does not accept
        raises there.
        """
        sanitized_kwargs: Dict[str, Any] = dict(kwargs)
        if namespace is not None:
            sanitized_kwargs.setdefault("namespace", namespace)
        if source_order is not None:
            sanitized_kwargs.setdefault("source_order", source_order)
        return sanitized_kwargs

    def param(self, path: str, **kwargs: Any) -> Any:
        """Retrieve a required parameter via dot-delimited path with validation."""
        options = {"validate": True, "required": True}
        options.update(kwargs)
        getter = getattr(self, "get_parameter_value", None)
        if not callable(getter):
            raise AttributeError("BridgeParameterMixin requires get_parameter_value on host class.")
        return getter(path, **options)

    def require_param(
        self,
        name: str,
        *,
        namespace: Optional[str] = None,
        units: Optional[str] = None,
        bounds: Optional[Tuple[float, float]] = None,
        validate: bool = True,
        expected_type: Optional[type] = None,
        source_order: Tuple[str, ...] = ("manager",),
    ) -> Any:
        """Retrieve a required parameter using manager-first lookup."""
        getter = getattr(self, "get_parameter_value", None)
        if not callable(getter):
            raise AttributeError("BridgeParameterMixin requires get_parameter_value on host class.")
        value = getter(
            name,
            namespace=namespace,
            units=units,
            bounds=bounds,
            validate=validate,
            expected_type=expected_type,
            source_order=source_order,
        )
        if value is None:
            full_key = f"{namespace}.{name}" if namespace else name
            raise KeyError(f"Missing required parameter '{full_key}'")
        return value

    def require_scalar(
        self,
        name: str,
        *,
        namespace: Optional[str] = None,
        units: Optional[str] = None,
        bounds: Optional[Tuple[float, float]] = None,
        validate: bool = True,
        expected_type: Optional[type] = None,
        source_order: Tuple[str, ...] = ("manager",),
    ) -> float:
        """Retrieve a required parameter and return a float scalar."""
        resolved_expected_type = expected_type
        if resolved_expected_type is None:
            resolved_expected_type = float
            resolve_lookup = getattr(self, "_resolve_parameter_lookup", None)
            manager_getter = getattr(self, "get_parameter_manager", None)
            if callable(resolve_lookup) and callable(manager_getter):
                full_key, _, _ = resolve_lookup(name, namespace)
                param_manager = manager_getter()
                if param_manager is not None:
                    spec = param_manager.get_parameter(full_key)
                    if spec is not None and getattr(spec, "requires_grad", False):
                        resolved_expected_type = None
        value = self.require_param(
            name,
            namespace=namespace,
            units=units,
            bounds=bounds,
            validate=validate,
            expected_type=resolved_expected_type,
            source_order=source_order,
        )
        if isinstance(value, Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def to_tensor(
        self,
        value: Any,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Convert a value to a tensor and align device."""
        tensor = _ensure_tensor(
            value,
            device=getattr(self, "device", None),
            dtype=dtype if dtype is not None else torch.float32,
        )
        target_device = device or getattr(self, "device", None)
        if target_device is not None and tensor.device != target_device:
            return tensor.to(target_device)
        return tensor

    def param_tensor(
        self,
        path: str,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Retrieve a parameter and return as a tensor."""
        value = self.param(path, **kwargs)
        return _ensure_tensor(value, device or getattr(self, "device", None)).to(dtype=dtype)

    def _sync_conservation_targets_from_metadata(self) -> None:
        """
        Load conservation ledger pools declared in metadata via ParameterManager.
        """
        metadata = getattr(self, "metadata", None)
        self._conservation_targets = {}
        if metadata is None or not getattr(metadata, "conservation_laws", None):
            return

        manager = _get_parameter_manager()  # type: ignore[no-untyped-call,unused-ignore]
        if manager is None:
            return

        targets: Dict[str, List[str]] = {}
        for law in metadata.conservation_laws:
            key = f"validation.conservation.ledger.{law}"
            # None means no ledger entry for this law; lookup failures propagate.
            pools = manager.get(key)
            if isinstance(pools, (list, tuple)):
                targets[law] = [str(pool) for pool in pools]

        self._conservation_targets = targets

    @property
    def conservation_targets(self) -> Dict[str, List[str]]:
        """
        Expose resolved conservation ledger targets for downstream diagnostics.
        """
        return getattr(self, "_conservation_targets", {})


class NamespaceDefaultMixin:
    """
    Mixin for functors whose parameter lookups share one class namespace and
    source order; every other lookup option is forwarded unchanged.
    """

    parameter_lookup_namespace: Optional[str] = None
    parameter_lookup_source_order: Optional[Tuple[str, ...]] = None

    def get_parameter_value(self, name: str, **kwargs: Any) -> Any:
        normalizer = getattr(self, "normalize_parameter_lookup_kwargs", None)
        if not callable(normalizer):
            raise AttributeError(
                "NamespaceDefaultMixin requires BridgeParameterMixin.normalize_parameter_lookup_kwargs "
                "on the host class."
            )
        sanitized_kwargs = normalizer(
            kwargs,
            namespace=self.parameter_lookup_namespace,
            source_order=self.parameter_lookup_source_order,
        )
        parent_getter = getattr(super(), "get_parameter_value", None)
        if not callable(parent_getter):
            raise AttributeError("NamespaceDefaultMixin requires superclass get_parameter_value.")
        return parent_getter(name=name, **sanitized_kwargs)


__all__ = [
    "BridgeParameterMixin",
    "NamespaceDefaultMixin",
    "_ensure_tensor",
]
