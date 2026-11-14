"""
Shared utilities for internal coupling modules.

Provides:
- CouplingParameterMixin: consistent parameter access across internal couplers
"""

from typing import Any, Dict, Optional, List, Iterable, Sequence, Tuple

import torch
from relic.core import device_policy
from torch import Tensor

from relic.parameters.namespaces import resolve_namespace
from relic.coupling.internal.dependency_utils import resolve_dependencies


class CouplingParameterMixin:
    """
    Helper mixin for internal coupling functors to retrieve namespaced parameters
    with consistent validation and typing.
    """

    def _get_coupling_parameter(
        self,
        key: str,
        *,
        namespace: Any,
        units: Optional[str] = None,
        required: bool = True,
        as_tensor: bool = False,
        as_scalar: bool = False,
        expected_type: Optional[type] = None,
    ) -> Any:
        """
        Retrieve a parameter from the ParameterManager with uniform typing.

        An absent optional key (required=False) is returned as None by
        get_parameter_value itself; any exception (unit, type or validation error)
        propagates for required and optional parameters alike.
        """
        resolved_ns = resolve_namespace(namespace)
        value = self.get_parameter_value(
            key,
            namespace=resolved_ns,
            units=units,
            required=required,
            expected_type=expected_type,
            validate=True,
        )

        # get_parameter_value returns a Tensor for numerics; convert explicitly.
        if value is not None:
            if as_scalar:
                if isinstance(value, Tensor):
                    value = value.detach().cpu().item()
            elif as_tensor:
                if not isinstance(value, Tensor):
                    device = device_policy.resolve_device(
                        explicit_device=getattr(self, "device", None),
                        module_name=self.__class__.__module__,
                        default="auto",
                    )
                    value = torch.tensor(float(value), device=device, dtype=torch.float32)
        return value


    def _initialize_coupling_dependencies(
        self,
        *,
        current_dependencies: Optional[Iterable[Any]],
        config: Optional[Dict[str, Any]],
        default_target: str,
        override_keys: Sequence[str],
    ) -> Tuple[List[str], bool]:
        """
        Resolve, validate, and persist coupling dependencies in one place.

        The resolved names become the FunctorBase instance declaration, which
        get_dependencies() reports as required edges (the single declaration
        channel read by the simulator and the DependencyGraphResolver).

        Returns
        -------
        Tuple[List[str], bool]
            Final dependency list and whether dependencies were explicitly defined by user input.
        """
        explicit_dependencies, explicit_from_user = resolve_dependencies(
            current_dependencies=current_dependencies,
            config=config,
            default_target=default_target,
            override_keys=override_keys,
        )

        self.set_dependencies(explicit_dependencies)

        return explicit_dependencies, explicit_from_user
