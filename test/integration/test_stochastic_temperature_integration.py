"""GraphIR and StateManager integration of the Ornstein-Uhlenbeck anomaly Functor.

References:
    - Hasselmann (1976), Tellus 28, 473-485.
    - Kloeden and Platen (1992), ch. 10.
"""

import pytest
import torch

from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.scientific.atmosphere.stochastic_temperature import (
    OrnsteinUhlenbeckTemperatureFunctor,
)
from relic.stochastic.workflows import build_bound_continuous_sde_graph

STATE = "stochastic_air_temperature_anomaly"
PROCESS = "stochastic_air_temperature"
FUNCTOR_CLASS = (
    "relic.scientific.atmosphere.stochastic_temperature."
    "OrnsteinUhlenbeckTemperatureFunctor"
)


def _graph(seed):
    """The production graph owner used by the public API/CLI builds the graph."""
    bound = build_bound_continuous_sde_graph(PROCESS, seed=seed)
    return bound.registry, bound.graph


class TestStochasticTemperatureGraph:
    def test_strict_graph_compiles_with_the_executable_continuous_sde_binding(self):
        registry, graph = _graph(seed=7)
        assert graph.nodes[PROCESS].class_type == f"Functor:{FUNCTOR_CLASS}"
        assert graph.nodes[PROCESS].config == {"seed": 7}
        context = graph.assertion["context"]
        assert context["required_semantics"] == ["continuous_sde"]
        assert context["state_units"] == {STATE: "K"}
        binding = context["execution_bindings"]["continuous_sde"]
        assert binding["functor_class"] == FUNCTOR_CLASS
        assert binding["forward_owner"] == FUNCTOR_CLASS + ".forward"
        assert binding["solver_owner"] == "relic.stochastic.sde_solver.EulerMaruyamaSolver"
        assert binding["state_vars"] == [STATE]
        assert binding["rng_state_scope"] == "functor_extra_state"
        assert context["stochastic_terms"][0]["parameter_keys"] == [
            "stochastic.weather.temperature.reversion_rate",
            "stochastic.weather.temperature.volatility",
        ]
        plan = compile_graph(graph, registry, strict=True)
        assert plan.report.validation.errors == []
        audit = plan.report.validation.assertion_audit
        assert audit["present"] is True and audit["valid"] is True and audit["op_resolved"] is True
        assert audit["structural_result"]["passed"] is True

    def test_unbound_process_names_fail_closed(self):
        with pytest.raises(ValueError, match="bound continuous-SDE process"):
            build_bound_continuous_sde_graph("decomposition", seed=7)

    def test_missing_execution_binding_fails_closed_under_stochastic_extension(self):
        from relic.validation.framework.rules.stochastic_sdae import validate_stochastic_sdae_payload

        registry, graph = _graph(seed=7)
        graph.assertion["context"].pop("execution_bindings")
        with pytest.raises(ValueError, match="Stochastic IPHS/SDAE structure is incomplete"):
            compile_graph(graph, registry, strict=True)
        result = validate_stochastic_sdae_payload(
            graph.assertion["context"], required_semantics=("continuous_sde",)
        )
        assert result.passed is False
        assert any("drift-diffusion binding [N]" in detail.expected for detail in result.details)

    def test_executor_advances_the_governed_anomaly_with_the_functor_substream(self):
        registry, graph = _graph(seed=7)
        plan = compile_graph(graph, registry, strict=True)
        executor = GraphExecutor(plan, registry=registry)
        final = executor.execute({STATE: torch.tensor(1.0, dtype=torch.float64)}, n_steps=3, dt=3600.0)

        runtime = executor._simulator
        functor = runtime.functors[PROCESS]

        # The same seed on a standalone Functor on the same device reproduces
        # the governed path exactly (CPU and CUDA generators are distinct streams).
        reference = OrnsteinUhlenbeckTemperatureFunctor(
            name=PROCESS, config={"seed": 7}, device=functor.device
        )
        value = torch.tensor(1.0, dtype=torch.float64, device=functor.device)
        for step in range(3):
            value = (value + reference.forward({STATE: value}, step * 3600.0, 3600.0)[STATE].value).detach()
        torch.testing.assert_close(final[STATE].detach().cpu().reshape(()), value.cpu(), rtol=0.0, atol=0.0)
        assert functor.get_extra_state()["draw_count"] == 3
        provenance = runtime.get_variable_provenance(STATE)
        assert provenance[-1]["functor"] == PROCESS
        assert provenance[-1]["role"] == "final_writer"
