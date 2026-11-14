"""
Common coupling-layer mocks shared across coupling integration tests.
"""

from typing import Any, Dict, Optional

import torch

from relic.core.functor import (
    BiologicalProcessFunctor,
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
)

__all__ = ["DummyPhotosynthesis"]


# References: Farquhar et al. 1980 (CO2 assimilation stub); Expression: A_net = net_assimilation_rate (default 0.0);
# Parameters: net_assimilation_rate (kg C m-2 s-1) comes from state to satisfy dependency ordering only.
class DummyPhotosynthesis(BiologicalProcessFunctor):
    """Minimal photosynthesis stub to satisfy dependency graph ordering."""

    def __init__(self, name: str = "photosynthesis_stub", device: Optional[torch.device] = None) -> None:
        metadata = FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Stub photosynthesis for dependency resolution",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            input_variables=["co2_concentration"],
            output_variables=["net_assimilation_rate"],
            references=["test_stub_photosynthesis"],
            mathematical_expression="A_net = 0",
        )
        super().__init__(name=name, config={}, metadata=metadata)
        self._device = device or torch.device("cpu")

    def get_dependencies(self):
        return []

    def forward(self, state: Dict[str, Any], time: Any, dt: Any, **kwargs):
        target_device = self._device
        state_value = state.get("net_assimilation_rate")
        if isinstance(state_value, torch.Tensor):
            target_device = state_value.device
        elif isinstance(state.get("co2_concentration"), torch.Tensor):
            target_device = state["co2_concentration"].device
        default = torch.tensor(0.0, device=target_device)
        result = state_value if isinstance(state_value, torch.Tensor) else default
        return {"net_assimilation_rate": result}


