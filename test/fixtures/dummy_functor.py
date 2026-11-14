"""Dummy functor helpers for orchestration tests."""

from typing import Any, Mapping

import torch
from torch import Tensor

from relic.core.functor import (
    BiologicalProcessFunctor,
    ExecutionPhase,
    FunctorMetadata,
    PhysicalFieldFunctor,
    ProcessCategory,
)


class DummyIncrementFunctor(BiologicalProcessFunctor):
    """Minimal functor that increments state variable x by 1 each step."""

    def __init__(
        self,
        name: str = "dummy_increment",
        config: dict[str, Any] | None = None,
    ) -> None:
        resolved_config = dict(config or {})
        configured_input = resolved_config.get("input_name")
        input_variables = (
            [] if configured_input in {None, ""} else [str(configured_input)]
        )
        metadata = FunctorMetadata(
            name=name,
            description="Dummy increment functor for orchestration tests.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="x_{t+1} = x_t + 1",
            references=["internal-test"],
            parameters={},
            input_variables=input_variables,
            output_variables=["x"],
        )
        super().__init__(
            name=name,
            config=resolved_config,
            metadata=metadata,
        )

    def get_dependencies(self) -> list[str]:
        return []

    def forward(self, state: dict[str, Tensor], time: float, dt: float, **kwargs: Any) -> dict[str, Tensor]:
        # References: internal-test
        # Equation: x_{t+1} = x_t + 1
        # Parameters: none
        return {"x": torch.tensor(1.0)}


class DummyIPHSFunctor(DummyIncrementFunctor):
    """Minimal protocol-complete Functor for compiler identity tests."""

    def declared_forms(self) -> tuple[object, ...]:
        return ()

    def required_parameters(self) -> tuple[str, ...]:
        return ()

    def build_hodge_terms(
        self,
        complex_bundle: Any,
        resolved_parameters: Mapping[str, Any],
    ) -> tuple[object, ...]:
        _ = complex_bundle, resolved_parameters
        return ()

    def energy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        _ = state
        return {}

    def entropy_terms(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        _ = state
        return {}

    def reversible_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any,
    ) -> Mapping[str, Any]:
        _ = state, complex_bundle
        return {}

    def dissipative_structure(
        self,
        state: Mapping[str, Any],
        complex_bundle: Any,
    ) -> Mapping[str, Any]:
        _ = state, complex_bundle
        return {}

    def ports(self) -> tuple[object, ...]:
        return ()

    def compute(
        self,
        state: Mapping[str, Any],
        time: float,
        dt: float,
    ) -> Mapping[str, Any]:
        return self.forward(dict(state), time=time, dt=dt)


class DummyPureFieldIncrementFunctor(PhysicalFieldFunctor):
    """Deterministic stateless field increment for cache tests."""

    cache_classification = "deterministic_pure"

    # Point model: no coordinate query; the state is read from StateManager.
    SPATIAL_DIMS = None

    def get_value_at(self, coordinates: "torch.Tensor") -> "torch.Tensor":
        """No coordinate query: this Functor is a point model (SPATIAL_DIMS = None)."""
        raise NotImplementedError(
            f"{type(self).__name__} is a point model (SPATIAL_DIMS = None): it has no "
            "coordinate query; read its state from StateManager"
        )

    def __init__(
        self,
        name: str = "dummy_pure_field_increment",
        config: dict[str, Any] | None = None,
    ) -> None:
        resolved_config = dict(config or {})
        input_name = str(resolved_config.get("input_name", "x"))
        metadata = FunctorMetadata(
            name=name,
            description="Stateless field increment for orchestration cache tests.",
            category=ProcessCategory.PHYSICAL_FIELD,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="Delta x = 1",
            references=["RELIC internal cache-contract verification"],
            parameters={},
            input_variables=[input_name],
            output_variables=["x"],
        )
        super().__init__(
            name=name,
            config=resolved_config,
            metadata=metadata,
        )

    def get_dependencies(self) -> list[str]:
        return []

    def forward(
        self,
        state: dict[str, Tensor],
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        # References: RELIC internal cache-contract verification.
        # Equation: Delta x = 1.
        # Parameters: x defines output device and dtype; no model parameter is used.
        _ = time, dt, kwargs
        return {"x": torch.ones_like(state["x"])}


class DummySinkFunctor(BiologicalProcessFunctor):
    """Minimal dependency sink with no state writer."""

    def __init__(
        self,
        name: str = "dummy_sink",
        config: dict[str, Any] | None = None,
    ) -> None:
        resolved_config = dict(config or {})
        input_name = str(resolved_config.get("input_name", "converted_x"))
        metadata = FunctorMetadata(
            name=name,
            description="Dummy dependency sink for orchestration tests.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="y = identity(x)",
            references=["internal-test"],
            parameters={},
            input_variables=[input_name],
            output_variables=[],
        )
        super().__init__(
            name=name,
            config=resolved_config,
            metadata=metadata,
        )

    def get_dependencies(self) -> list[str]:
        return []

    def forward(
        self,
        state: dict[str, Tensor],
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        # References: internal-test.
        # Equation: the sink observes x and creates no state update.
        # Parameters: none.
        _ = state, time, dt, kwargs
        return {}


class DummyConfiguredOutputFunctor(BiologicalProcessFunctor):
    """Minimal functor with one config-selected state output."""

    def __init__(
        self,
        name: str = "dummy_configured_output",
        config: dict[str, Any] | None = None,
    ) -> None:
        resolved_config = dict(config or {})
        output_name = str(resolved_config.get("output_name", "x"))
        metadata = FunctorMetadata(
            name=name,
            description="Dummy configurable writer for orchestration tests.",
            category=ProcessCategory.BIOLOGICAL_PROCESS,
            execution_phase=ExecutionPhase.MAIN_PROCESSES,
            mathematical_expression="y = 1",
            references=["internal-test"],
            parameters={},
            input_variables=[],
            output_variables=[output_name],
        )
        super().__init__(
            name=name,
            config=resolved_config,
            metadata=metadata,
        )
        self.output_name = output_name

    def get_dependencies(self) -> list[str]:
        return []

    def forward(
        self,
        state: dict[str, Tensor],
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        # References: internal-test.
        # Equation: y = 1.
        # Parameters: output_name selects the state key.
        _ = state, time, dt, kwargs
        return {self.output_name: torch.tensor(1.0)}
