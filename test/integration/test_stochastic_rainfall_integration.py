"""GraphIR and StateManager integration of the marked-Poisson rainfall Functor.

References:
    - Rodriguez-Iturbe et al. (1999), Proc. R. Soc. A 455, 3789-3805.
    - Gillespie (1977), J. Phys. Chem. 81, 2340-2361.
"""

import pytest
import torch

from relic.orchestration.compiler import compile_graph
from relic.orchestration.executor import GraphExecutor
from relic.scientific.atmosphere.stochastic_rainfall import MarkedPoissonRainfallFunctor
from relic.stochastic.workflows import build_bound_marked_jump_graph

COUNT = "stochastic_rainfall_event_count"
DEPTH = "stochastic_rainfall_depth"
PROCESS = "stochastic_rainfall"
FUNCTOR_CLASS = "relic.scientific.atmosphere.stochastic_rainfall.MarkedPoissonRainfallFunctor"


def _graph(seed):
    """The production graph owner used by the public API/CLI builds the graph."""
    bound = build_bound_marked_jump_graph(PROCESS, seed=seed)
    return bound.registry, bound.graph


class TestStochasticRainfallGraph:
    def test_strict_graph_compiles_with_the_executable_marked_jump_binding(self):
        registry, graph = _graph(seed=7)
        assert graph.nodes[PROCESS].class_type == f"Functor:{FUNCTOR_CLASS}"
        assert graph.nodes[PROCESS].config == {"seed": 7}
        context = graph.assertion["context"]
        assert context["required_semantics"] == ["marked_jump"]
        assert context["state_units"] == {COUNT: "1", DEPTH: "mm"}
        jump = context["jump_resets"][0]
        assert jump["reset_map"] == [[1.0]]
        assert jump["ledger_behavior"] == "open" and jump["open_system_route"] == "port"
        assert jump["propensity_keys"] == ["stochastic.weather.rainfall.event_rate"]
        assert "stochastic.weather.rainfall.mean_depth" in jump["mark_space"]
        assert jump["event_clock"] == "direct_ssa_fixed_step"
        assert context["stochastic_reproducibility"]["noise_type"] == "marked_poisson"
        assert context["stochastic_reproducibility"]["solver_method"] == "direct_ssa"
        binding = context["execution_bindings"]["marked_jump"]
        assert binding["functor_class"] == FUNCTOR_CLASS
        assert binding["solver_owner"] == "relic.stochastic.gillespie.GillespieSSA"
        assert binding["state_vars"] == [COUNT, DEPTH]
        assert binding["rng_state_scope"] == "functor_extra_state"
        plan = compile_graph(graph, registry, strict=True)
        assert plan.report.validation.errors == []
        audit = plan.report.validation.assertion_audit
        assert audit["present"] is True and audit["valid"] is True and audit["op_resolved"] is True
        assert audit["structural_result"]["passed"] is True

    def test_missing_execution_binding_fails_closed_under_stochastic_extension(self):
        from relic.validation.framework.rules.stochastic_sdae import validate_stochastic_sdae_payload

        registry, graph = _graph(seed=7)
        graph.assertion["context"].pop("execution_bindings")
        with pytest.raises(ValueError, match="Stochastic IPHS/SDAE structure is incomplete"):
            compile_graph(graph, registry, strict=True)
        result = validate_stochastic_sdae_payload(
            graph.assertion["context"], required_semantics=("marked_jump",)
        )
        assert result.passed is False
        assert any("marked-jump execution binding [N]" in detail.expected for detail in result.details)

    def test_unbound_process_names_fail_closed(self):
        with pytest.raises(ValueError, match="bound marked-jump process"):
            build_bound_marked_jump_graph("stochastic_air_temperature", seed=7)

    def test_executor_advances_the_governed_counts_with_the_functor_substream(self):
        registry, graph = _graph(seed=7)
        plan = compile_graph(graph, registry, strict=True)
        executor = GraphExecutor(plan, registry=registry)
        zeros = torch.zeros(4, dtype=torch.float64)
        final = executor.execute({COUNT: zeros.clone(), DEPTH: zeros.clone()}, n_steps=3, dt=6.0 * 3600.0)

        runtime = executor._simulator
        functor = runtime.functors[PROCESS]

        reference = MarkedPoissonRainfallFunctor(name=PROCESS, config={"seed": 7}, device=functor.device)
        values = {
            COUNT: torch.zeros(4, dtype=torch.float64, device=functor.device),
            DEPTH: torch.zeros(4, dtype=torch.float64, device=functor.device),
        }
        for step in range(3):
            outputs = reference.forward(dict(values), step * 6.0 * 3600.0, 6.0 * 3600.0)
            for name in values:
                values[name] = (values[name] + outputs[name].value).detach()
        for name in (COUNT, DEPTH):
            torch.testing.assert_close(final[name].detach().cpu(), values[name].cpu(), rtol=0.0, atol=0.0)
        assert torch.equal(final[COUNT].detach().cpu(), torch.round(final[COUNT].detach().cpu()))
        assert functor.get_extra_state()["draw_count"] == 3
        provenance = runtime.get_variable_provenance(DEPTH)
        assert provenance[-1]["functor"] == PROCESS
        assert provenance[-1]["role"] == "final_writer"
