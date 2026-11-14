"""
L1 Unit Tests for Dependency Graph Resolver

Tests dependency graph functionality including:
- Basic dependency resolution and topological sorting
- Optional dependencies handling
- Cycle detection (using networkx)
- Dynamic functor addition/removal
- Parallel execution groups

Test Level: L1 (Unit)
Dependencies: DependencyGraphResolver, networkx
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from relic.core.dependency_graph import (
    DependencyGraphResolver,
    DependencyType,
    CyclicDependencyError,
    MissingDependencyError,
)


# Test functors
class SimpleFunctor:
    """Simple functor for testing"""
    def __init__(self, name, dependencies=None):
        self.name = name
        self._dependencies = dependencies or []

    def get_dependencies(self):
        return self._dependencies

    def forward(self, state, time, dt):
        return {f"{self.name}_output": torch.tensor(1.0)}


class StructuredDependencyFunctor:
    """Functor with structured dependencies (required/optional)"""
    def __init__(self, name, required_deps=None, optional_deps=None):
        self.name = name
        self._required = required_deps or []
        self._optional = optional_deps or []

    def get_dependencies(self):
        return {
            'required': self._required,
            'optional': self._optional
        }

    def forward(self, state, time, dt):
        return {f"{self.name}_output": torch.tensor(2.0)}


def _resolver_state_signature(resolver):
    nx_nodes = ()
    nx_edges = ()
    if resolver._nx_graph is not None:
        nx_nodes = tuple(sorted(resolver._nx_graph.nodes))
        nx_edges = tuple(
            sorted(
                (
                    source,
                    target,
                    tuple(sorted((key, repr(value)) for key, value in data.items())),
                )
                for source, target, data in resolver._nx_graph.edges(data=True)
            )
        )
    return {
        "nodes": {
            node_id: (
                id(node.functor),
                frozenset(node.dependencies),
                frozenset(node.lagged_dependencies),
                frozenset(node.dependents),
                node.in_degree,
                node.out_degree,
                node.execution_time,
                repr(node.metadata),
            )
            for node_id, node in resolver.nodes.items()
        },
        "edges": tuple(
            (
                edge.source,
                edge.target,
                edge.dependency_type,
                edge.weight,
                edge.required,
                repr(edge.metadata),
            )
            for edge in resolver.edges
        ),
        "execution_order": (
            tuple(resolver.execution_order)
            if resolver.execution_order is not None
            else None
        ),
        "adjacency": {
            key: frozenset(values)
            for key, values in resolver._adjacency_list.items()
        },
        "reverse_adjacency": {
            key: frozenset(values)
            for key, values in resolver._reverse_adjacency.items()
        },
        "optional_deps": {
            key: frozenset(values)
            for key, values in resolver._optional_deps.items()
        },
        "lagged_deps": {
            key: frozenset(values)
            for key, values in resolver._lagged_deps.items()
        },
        "nx_nodes": nx_nodes,
        "nx_edges": nx_edges,
    }


class TestBasicDependencyResolution:
    """Test basic dependency graph functionality"""

    def test_empty_graph_initialization(self):
        """Test creating an empty dependency graph"""
        resolver = DependencyGraphResolver()

        assert len(resolver.nodes) == 0
        assert len(resolver.edges) == 0
        assert resolver.execution_order is None

    def test_add_single_functor_no_dependencies(self):
        """Test adding a functor with no dependencies"""
        resolver = DependencyGraphResolver()
        functor = SimpleFunctor("test_functor")

        fid = resolver.add_functor(functor)

        assert fid == "test_functor"
        assert "test_functor" in resolver.nodes
        assert len(resolver.nodes["test_functor"].dependencies) == 0

    def test_add_functor_with_dependencies(self):
        """Test adding functor with dependencies"""
        resolver = DependencyGraphResolver()

        f1 = SimpleFunctor("base")
        f2 = SimpleFunctor("dependent", dependencies=["base"])

        resolver.add_functor(f1)
        resolver.add_functor(f2)

        assert "dependent" in resolver.nodes
        assert "base" in resolver.nodes["dependent"].dependencies

    def test_topological_sort_simple_chain(self):
        """Test topological sort with simple dependency chain"""
        resolver = DependencyGraphResolver()

        # A -> B -> C
        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.add_functor(SimpleFunctor("C", ["B"]))

        order = resolver.topological_sort()

        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_topological_sort_parallel_dependencies(self):
        """Test topological sort with parallel branches"""
        resolver = DependencyGraphResolver()

        # A <- B
        #   <- C <- D
        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.add_functor(SimpleFunctor("C", ["A"]))
        resolver.add_functor(SimpleFunctor("D", ["C"]))

        order = resolver.topological_sort()

        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("C") < order.index("D")

    def test_nodes_store_functor_instances(self):
        """Test canonical nodes mapping stores each functor instance."""
        resolver = DependencyGraphResolver()

        f1 = SimpleFunctor("test1")
        f2 = SimpleFunctor("test2")

        resolver.add_functor(f1)
        resolver.add_functor(f2)

        assert "test1" in resolver.nodes
        assert "test2" in resolver.nodes
        assert resolver.nodes["test1"].functor == f1
        assert resolver.nodes["test2"].functor == f2

    def test_topological_sort_is_stable(self):
        """Test repeated canonical topological sorting is stable."""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))

        order1 = resolver.topological_sort()
        order2 = resolver.topological_sort()

        assert order1 == order2


class TestCycleDetection:
    """Test cycle detection functionality"""

    def test_simple_cycle_detection(self):
        """Test detection of simple cycle: A -> B -> A"""
        resolver = DependencyGraphResolver()

        # Create cycle manually via add_edge
        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B"))

        resolver.add_edge("A", "B")
        resolver.add_edge("B", "A")

        with pytest.raises(CyclicDependencyError) as exc_info:
            resolver.topological_sort()

        assert "A" in exc_info.value.cycle
        assert "B" in exc_info.value.cycle

    def test_complex_cycle_detection(self):
        """Test detection of complex cycle: A -> B -> C -> A"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B"))
        resolver.add_functor(SimpleFunctor("C"))

        resolver.add_edge("A", "B")
        resolver.add_edge("B", "C")
        resolver.add_edge("C", "A")

        with pytest.raises(CyclicDependencyError):
            resolver.topological_sort()

    def test_cycle_search_without_a_cycle_is_an_inconsistency(self):
        """_find_cycle never invents a cycle from arbitrary nodes."""
        resolver = DependencyGraphResolver()
        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B"))
        resolver.topological_sort()

        with pytest.raises(RuntimeError, match="no cycle"):
            resolver._find_cycle({"A", "B"})

    def test_no_cycle_with_diamond_dependency(self):
        """Test that diamond dependencies are handled correctly (not mistaken for cycle)"""
        resolver = DependencyGraphResolver()

        # Diamond: A <- B, A <- C, B <- D, C <- D
        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.add_functor(SimpleFunctor("C", ["A"]))
        resolver.add_functor(SimpleFunctor("D", ["B", "C"]))

        order = resolver.topological_sort()

        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")


class TestSingleSortingPath:
    """One sorting path over the authoritative adjacency (Kahn, registration tie-break)."""

    def test_order_comes_from_the_declared_graph_not_the_networkx_mirror(self, monkeypatch):
        import relic.core.dependency_graph as dependency_graph

        def _forbidden(*_args, **_kwargs):
            raise AssertionError("the networkx mirror is not a sorting path")

        monkeypatch.setattr(dependency_graph.nx, "lexicographical_topological_sort", _forbidden)
        monkeypatch.setattr(dependency_graph.nx, "is_directed_acyclic_graph", _forbidden)
        resolver = DependencyGraphResolver(use_networkx=True)
        resolver.add_functor(SimpleFunctor("a"))
        resolver.add_functor(SimpleFunctor("b", ["a"]))
        resolver.add_functor(SimpleFunctor("c"))
        resolver._nx_graph.remove_node("b")

        assert resolver.topological_sort() == ["a", "b", "c"]

    @pytest.mark.parametrize("name", [None, "", 7])
    def test_functor_without_a_name_is_rejected(self, name):
        class _Anonymous:
            def get_dependencies(self):
                return []

        functor = _Anonymous()
        functor.name = name
        resolver = DependencyGraphResolver()
        with pytest.raises(ValueError, match="name"):
            resolver.add_functor(functor)

    def test_functor_without_a_name_attribute_is_rejected(self):
        class _Nameless:
            def get_dependencies(self):
                return []

        with pytest.raises(ValueError, match="name"):
            DependencyGraphResolver().add_functor(_Nameless())

    def test_metadata_conversion_failures_propagate(self):
        class _BrokenMetadata:
            def to_dict(self):
                raise RuntimeError("metadata cannot be serialized")

        with pytest.raises(RuntimeError, match="metadata cannot be serialized"):
            DependencyGraphResolver._extract_networkx_metadata_payload(_BrokenMetadata())
        with pytest.raises(TypeError, match="metadata"):
            DependencyGraphResolver._extract_networkx_metadata_payload(5)
        assert DependencyGraphResolver._extract_networkx_metadata_payload(None) == {}


@pytest.mark.parametrize("use_networkx", [False, True])
class TestOptionalDependencies:
    """Test optional dependency handling"""

    def test_optional_dependency_with_missing_target(self, use_networkx):
        """Test that missing optional dependencies are gracefully skipped"""
        resolver = DependencyGraphResolver(use_networkx=use_networkx)

        # Create functor with optional dependency that doesn't exist
        functor = StructuredDependencyFunctor(
            "test",
            required_deps=["base"],
            optional_deps=["missing_optional"]
        )

        resolver.add_functor(SimpleFunctor("base"))
        resolver.add_functor(functor)

        # Should not raise error
        order = resolver.topological_sort()

        assert "base" in order
        assert "test" in order
        assert order.index("base") < order.index("test")

    def test_optional_dependency_with_existing_target(self, use_networkx):
        """Test that existing optional dependencies are included"""
        resolver = DependencyGraphResolver(use_networkx=use_networkx)

        functor = StructuredDependencyFunctor(
            "test",
            required_deps=["base"],
            optional_deps=["optional_exists"]
        )

        resolver.add_functor(functor)
        resolver.add_functor(SimpleFunctor("optional_exists", ["base"]))
        resolver.add_functor(SimpleFunctor("base"))

        order = resolver.topological_sort()

        assert all(f in order for f in ["base", "optional_exists", "test"])
        assert order.index("base") < order.index("test")
        assert order.index("optional_exists") < order.index("test")

    def test_optional_dependency_type_in_edge(self, use_networkx):
        """Test that optional dependencies can be explicitly marked"""
        resolver = DependencyGraphResolver(use_networkx=use_networkx)

        resolver.add_functor(SimpleFunctor("B"))
        resolver.add_functor(SimpleFunctor("A"))

        # Add optional edge
        resolver.add_edge("B", "A", dependency_type=DependencyType.OPTIONAL)

        # Should work even if A is missing (for this test, A exists)
        order = resolver.topological_sort()

        assert "A" in order
        assert "B" in order
        assert order.index("A") < order.index("B")

    def test_mixed_required_and_optional_dependencies(self, use_networkx):
        """Test functor with both required and optional dependencies"""
        resolver = DependencyGraphResolver(use_networkx=use_networkx)

        functor = StructuredDependencyFunctor(
            "mixed",
            required_deps=["required1", "required2"],
            optional_deps=["opt1", "opt2_missing"]
        )

        resolver.add_functor(SimpleFunctor("required1"))
        resolver.add_functor(SimpleFunctor("required2"))
        resolver.add_functor(SimpleFunctor("opt1"))
        # opt2_missing intentionally not added
        resolver.add_functor(functor)

        order = resolver.topological_sort()

        assert all(f in order for f in ["required1", "required2", "opt1", "mixed"])
        # Required dependencies must come before mixed
        assert order.index("required1") < order.index("mixed")
        assert order.index("required2") < order.index("mixed")
        # Existing optional dependency must come before mixed
        assert order.index("opt1") < order.index("mixed")


    @pytest.mark.parametrize("self_cycle", [False, True])
    def test_existing_optional_cycle_is_rejected(self, use_networkx, self_cycle):
        resolver = DependencyGraphResolver(use_networkx=use_networkx)
        resolver.add_functor(
            StructuredDependencyFunctor(
                "A", optional_deps=["A" if self_cycle else "B"]
            )
        )
        if not self_cycle:
            resolver.add_functor(StructuredDependencyFunctor("B", optional_deps=["A"]))

        with pytest.raises(CyclicDependencyError) as error:
            resolver.topological_sort()

        assert set(error.value.cycle) == ({"A"} if self_cycle else {"A", "B"})
        assert resolver.execution_order is None

    def test_lagged_feedback_is_not_a_same_step_dependency(self, use_networkx):
        resolver = DependencyGraphResolver(use_networkx=use_networkx)
        resolver.add_functor(
            SimpleFunctor("consumer", {"optional": ["provider", "missing"]})
        )
        resolver.add_functor(
            SimpleFunctor("provider", {"lagged": ["consumer", "absent_history"]})
        )

        assert resolver.topological_sort() == ["provider", "consumer"]
        assert resolver.get_parallel_groups() == [["provider"], ["consumer"]]
        assert resolver.nodes["provider"].lagged_dependencies == {
            "consumer", "absent_history"
        }

    def test_rebuild_counts_each_resolved_edge_once(self, use_networkx):
        resolver = DependencyGraphResolver(use_networkx=use_networkx)
        resolver.add_functor(StructuredDependencyFunctor("consumer", optional_deps=["provider"]))
        resolver.add_functor(SimpleFunctor("provider"))
        resolver.add_edge("consumer", "provider", required=False)
        resolver.add_edge("consumer", "provider", required=False)

        for _ in range(2):
            resolver.execution_order = None
            assert resolver.topological_sort() == ["provider", "consumer"]
            assert len(resolver.edges) == 1
            assert resolver.nodes["consumer"].out_degree == 1
            assert resolver.nodes["provider"].in_degree == 1

    @pytest.mark.parametrize("warm_cache", [False, True])
    def test_missing_optional_can_be_required_by_policy(self, use_networkx, warm_cache):
        resolver = DependencyGraphResolver(use_networkx=use_networkx)
        resolver.add_functor(StructuredDependencyFunctor("consumer", optional_deps=["missing"]))
        if warm_cache:
            assert resolver.topological_sort() == ["consumer"]

        with pytest.raises(MissingDependencyError):
            resolver.topological_sort(handle_optional=False)


class TestDynamicDependencyResolution:
    """Test dynamic functor addition and removal"""

    @pytest.mark.parametrize("dynamic", [False, True])
    def test_duplicate_functor_id_is_rejected_without_mutation(self, dynamic):
        """Duplicate IDs must not act as an undeclared hot-swap path."""
        resolver = DependencyGraphResolver(use_networkx=True)
        original = SimpleFunctor("shared")
        resolver.add_functor(original)
        resolver.topological_sort()
        before = _resolver_state_signature(resolver)

        add = resolver.add_functor_dynamic if dynamic else resolver.add_functor
        with pytest.raises(ValueError, match="already exists"):
            add(SimpleFunctor("shared"))

        assert _resolver_state_signature(resolver) == before
        assert resolver.nodes["shared"].functor is original

    def test_add_functor_dynamic_basic(self):
        """Test dynamically adding a functor"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("base"))
        order1 = resolver.topological_sort()

        # Dynamically add new functor
        fid = resolver.add_functor_dynamic(SimpleFunctor("dynamic"))

        assert fid == "dynamic"
        assert "dynamic" in resolver.nodes
        # Execution order should be rebuilt
        assert resolver.execution_order is not None

    def test_add_functor_dynamic_with_dependencies(self):
        """Test dynamically adding a functor with dependencies"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.topological_sort()

        # Add C that depends on B
        resolver.add_functor_dynamic(SimpleFunctor("C", ["B"]))

        order = resolver.execution_order

        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_add_functor_dynamic_creates_cycle(self):
        """Test that adding a functor that creates a cycle is rejected"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.topological_sort()

        # Try to add functor that creates cycle: A -> B -> C -> A
        # We need to manually create this scenario
        resolver.add_functor(SimpleFunctor("C", ["B"]))
        resolver.add_edge("A", "C")  # This creates cycle

        with pytest.raises(CyclicDependencyError):
            resolver.add_functor_dynamic(SimpleFunctor("dummy"), rebuild_graph=True)

    def test_dynamic_add_missing_required_restores_entire_graph(self):
        resolver = DependencyGraphResolver(use_networkx=True)
        resolver.add_functor(SimpleFunctor("base"))
        resolver.topological_sort()
        before = _resolver_state_signature(resolver)

        with pytest.raises(MissingDependencyError):
            resolver.add_functor_dynamic(
                SimpleFunctor("bad", ["missing_provider"]),
                rebuild_graph=True,
            )

        assert _resolver_state_signature(resolver) == before

    def test_dynamic_cycle_rollback_clears_optional_dependency_state(self):
        resolver = DependencyGraphResolver(use_networkx=True)
        resolver.add_functor(
            StructuredDependencyFunctor("A", optional_deps=["C"])
        )
        resolver.topological_sort()
        before = _resolver_state_signature(resolver)

        with pytest.raises(CyclicDependencyError):
            resolver.add_functor_dynamic(
                StructuredDependencyFunctor(
                    "C",
                    required_deps=["A"],
                    optional_deps=["X"],
                ),
                rebuild_graph=True,
            )

        assert _resolver_state_signature(resolver) == before
        resolver.add_functor_dynamic(SimpleFunctor("X"))
        resolver.add_functor_dynamic(
            StructuredDependencyFunctor("C", required_deps=["X"])
        )
        matching_edges = [
            edge
            for edge in resolver.edges
            if edge.source == "C" and edge.target == "X"
        ]
        assert len(matching_edges) == 1
        assert matching_edges[0].required is True

    def test_remove_functor_basic(self):
        """Test removing a functor from graph"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B"))
        resolver.add_functor(SimpleFunctor("C"))

        resolver.remove_functor("B")

        assert "B" not in resolver.nodes
        assert "A" in resolver.nodes
        assert "C" in resolver.nodes

    def test_remove_functor_with_dependents(self):
        """Required providers must fail before graph mutation."""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.topological_sort()

        with pytest.raises(MissingDependencyError):
            resolver.remove_functor("A")

        assert "A" in resolver.nodes
        assert "B" in resolver.nodes
        assert "A" in resolver.nodes["B"].dependencies
        assert resolver.execution_order == ["A", "B"]

    @pytest.mark.parametrize("use_networkx", [False, True])
    def test_remove_optional_provider_preserves_declaration_and_rebinds(self, use_networkx):
        """Optional declarations survive provider removal and later rebind."""
        resolver = DependencyGraphResolver(use_networkx=use_networkx)
        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(
            StructuredDependencyFunctor("B", optional_deps=["A"])
        )
        resolver.topological_sort()

        resolver.remove_functor("A")

        assert "A" not in resolver.nodes
        assert "A" in resolver.nodes["B"].dependencies
        assert "A" in resolver._optional_deps["B"]
        assert resolver.execution_order == ["B"]

        resolver.add_functor_dynamic(SimpleFunctor("A"))

        assert resolver.execution_order.index("A") < resolver.execution_order.index("B")
        assert any(
            edge.source == "B" and edge.target == "A" and not edge.required
            for edge in resolver.edges
        )

    def test_remove_functor_not_found(self):
        """Test that removing non-existent functor raises error"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))

        with pytest.raises(KeyError):
            resolver.remove_functor("nonexistent")


class TestParallelExecutionGroups:
    """Test parallel execution group identification"""

    def test_get_parallel_groups_linear(self):
        """Test parallel groups for linear dependency chain"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.add_functor(SimpleFunctor("C", ["B"]))

        groups = resolver.get_parallel_groups()

        # Should have 3 groups (one per functor)
        assert len(groups) == 3
        assert ["A"] in groups
        assert ["B"] in groups
        assert ["C"] in groups

    def test_get_parallel_groups_with_parallelism(self):
        """Test parallel groups with actual parallelism"""
        resolver = DependencyGraphResolver()

        # A <- B, A <- C (B and C can run in parallel)
        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.add_functor(SimpleFunctor("C", ["A"]))

        groups = resolver.get_parallel_groups()

        # Should have 2 groups
        assert len(groups) == 2
        # First group: A
        assert "A" in groups[0]
        # Second group: B and C (parallel)
        assert set(groups[1]) == {"B", "C"}

    def test_get_parallel_groups_complex(self):
        """Test parallel groups with complex dependencies"""
        resolver = DependencyGraphResolver()

        # Level 0: A, B (parallel)
        # Level 1: C (depends on A), D (depends on B)
        # Level 2: E (depends on C and D)
        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B"))
        resolver.add_functor(SimpleFunctor("C", ["A"]))
        resolver.add_functor(SimpleFunctor("D", ["B"]))
        resolver.add_functor(SimpleFunctor("E", ["C", "D"]))

        groups = resolver.get_parallel_groups()

        assert len(groups) == 3
        assert set(groups[0]) == {"A", "B"}
        assert set(groups[1]) == {"C", "D"}
        assert groups[2] == ["E"]

    def test_parallel_groups_are_stable(self):
        """Test repeated canonical parallel grouping is stable."""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))

        groups1 = resolver.get_parallel_groups()
        groups2 = resolver.get_parallel_groups()

        assert groups1 == groups2


class TestNetworkXIntegration:
    """Test networkx integration features"""

    def test_use_networkx_flag(self):
        """Test that networkx can be enabled/disabled"""
        # Default should use networkx if available
        resolver1 = DependencyGraphResolver()
        assert resolver1.use_networkx == True  # Assuming networkx is installed
        assert resolver1._nx_graph is not None

        # Explicitly disable
        resolver2 = DependencyGraphResolver(use_networkx=False)
        assert resolver2.use_networkx == False
        assert resolver2._nx_graph is None

    def test_topological_sort_with_networkx(self):
        """Test that networkx topological sort works"""
        resolver = DependencyGraphResolver(use_networkx=True)

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.add_functor(SimpleFunctor("C", ["B"]))

        order = resolver.topological_sort()

        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_cycle_detection_with_networkx(self):
        """Test that networkx cycle detection works"""
        resolver = DependencyGraphResolver(use_networkx=True)

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B"))

        resolver.add_edge("A", "B")
        resolver.add_edge("B", "A")

        with pytest.raises(CyclicDependencyError):
            resolver.topological_sort()


class TestGraphStatistics:
    """Test dependency graph statistics"""

    def test_get_execution_stats_basic(self):
        """Test basic execution statistics"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.add_functor(SimpleFunctor("C", ["B"]))

        stats = resolver.get_execution_stats()

        assert stats['node_count'] == 3
        assert stats['edge_count'] >= 2  # At least A->B and B->C
        assert stats['has_cycles'] is False
        assert stats['execution_depth'] == 3

    def test_get_execution_stats_with_parallelism(self):
        """Test execution statistics with parallel groups"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))
        resolver.add_functor(SimpleFunctor("C", ["A"]))

        stats = resolver.get_execution_stats()

        assert stats['max_parallel_width'] == 2  # B and C can run in parallel
        assert stats['execution_depth'] == 2  # A, then B+C

    def test_validate_dependencies(self):
        """Test dependency validation"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A", "missing"]))

        missing = resolver.validate_dependencies()

        assert len(missing) > 0
        assert any("missing" in m for m in missing)


class TestVisualization:
    """Test dependency graph visualization"""

    def test_visualize_simple_graph(self):
        """Test DOT visualization generation"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))

        dot_str = resolver.visualize()

        assert "digraph DependencyGraph" in dot_str
        assert '"A"' in dot_str
        assert '"B"' in dot_str
        assert "->" in dot_str

    def test_visualize_optional_dependencies(self):
        """Test that optional dependencies are shown with dashed lines"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B"))

        resolver.add_edge("B", "A", dependency_type=DependencyType.OPTIONAL)

        dot_str = resolver.visualize()

        assert "dashed" in dot_str  # Optional dependencies use dashed style


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_graph_topological_sort(self):
        """Test topological sort on empty graph"""
        resolver = DependencyGraphResolver()

        order = resolver.topological_sort()

        assert order == []

    def test_single_functor_topological_sort(self):
        """Test topological sort with single functor"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("alone"))

        order = resolver.topological_sort()

        assert order == ["alone"]

    def test_execution_order_caching(self):
        """Test that execution order is cached"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        resolver.add_functor(SimpleFunctor("B", ["A"]))

        order1 = resolver.topological_sort()
        order2 = resolver.topological_sort()

        # Should return cached result (same list object)
        assert order1 is order2

    def test_execution_order_cache_invalidation(self):
        """Test that execution order cache is invalidated on graph changes"""
        resolver = DependencyGraphResolver()

        resolver.add_functor(SimpleFunctor("A"))
        order1 = resolver.topological_sort()

        # Add new functor
        resolver.add_functor(SimpleFunctor("B"))

        # Cache should be invalidated
        assert resolver.execution_order is None


class TestExecutionOwnership:
    """DependencyGraphResolver owns topology, not Functor execution or state writes."""

    def test_dependency_graph_has_no_execution_entry_point(self):
        assert not hasattr(DependencyGraphResolver, "execute_in_order")


class DeclaredDependencyFunctor:
    """Functor returning one raw dependency declaration."""

    def __init__(self, name, declaration):
        self.name = name
        self.declaration = declaration

    def get_dependencies(self):
        return self.declaration


class TestSingleDependencyNormalizer:
    """One owner normalizes required / optional / lagged declarations (list = required)."""

    def test_declared_forms_normalize_to_one_structure(self):
        from relic.core.dependency_graph import normalize_dependency_declaration

        declared = normalize_dependency_declaration(
            {"required": ("b", "a"), "optional": ["c", "c"], "lagged": ["d"]}, owner="consumer"
        )
        assert declared.required == ("b", "a")
        assert declared.optional == ("c",)
        assert declared.lagged == ("d",)
        assert declared.same_step == ("b", "a", "c")
        assert declared.canonical() == {"required": ["a", "b"], "optional": ["c"], "lagged": ["d"]}
        listed = normalize_dependency_declaration(["x", "y"], owner="consumer")
        assert (listed.required, listed.optional, listed.lagged) == (("x", "y"), (), ())
        empty = normalize_dependency_declaration(None, owner="consumer")
        assert empty.canonical() == {"required": [], "optional": [], "lagged": []}

    @pytest.mark.parametrize(
        ("declaration", "error", "match"),
        [
            ({"requird": ["a"]}, ValueError, "unknown dependency categories"),
            ({"provider_a": "optional"}, ValueError, "unknown dependency categories"),
            ({1: ["a"]}, ValueError, "unknown dependency categories"),
            ({"required": [1]}, TypeError, "non-string"),
            ({"optional": ["ok", None]}, TypeError, "non-string"),
            ({"required": ["  "]}, ValueError, "empty"),
            ({"required": "a"}, TypeError, "list or tuple"),
            ({"required": ["a"], "lagged": ["a"]}, ValueError, "more than one category"),
            ({"optional": ["b"], "lagged": ["b"]}, ValueError, "more than one category"),
            ("a", TypeError, "list, tuple, or mapping"),
            ({"a", "b"}, TypeError, "list, tuple, or mapping"),
        ],
    )
    def test_invalid_declarations_raise_in_the_normalizer_and_the_resolver(
        self, declaration, error, match
    ):
        from relic.core.dependency_graph import normalize_dependency_declaration

        with pytest.raises(error, match=match):
            normalize_dependency_declaration(declaration, owner="consumer")
        resolver = DependencyGraphResolver()
        with pytest.raises(error, match=match):
            resolver.add_functor(DeclaredDependencyFunctor("consumer", declaration))
        assert resolver.nodes == {}

    def test_optionality_comes_only_from_get_dependencies(self):
        """The resolver and the simulator read one declaration: get_dependencies().

        An ``optional_dependencies`` instance attribute is not a second channel;
        a name it lists does not become an edge, and optionality is declared as
        the ``optional`` category of the declaration.
        """
        resolver = DependencyGraphResolver()
        resolver.add_functor(SimpleFunctor("provider"))
        stray = DeclaredDependencyFunctor("stray_consumer", [])
        stray.optional_dependencies = ["provider"]
        resolver.add_functor(stray)
        assert resolver.nodes["stray_consumer"].dependencies == set()
        assert not resolver._optional_deps.get("stray_consumer")

        resolver.add_functor(
            DeclaredDependencyFunctor("declared_consumer", {"optional": ["provider"]})
        )
        assert resolver.nodes["declared_consumer"].dependencies == {"provider"}
        assert resolver._optional_deps["declared_consumer"] == {"provider"}

    def test_resolver_reads_tuple_categories(self):
        resolver = DependencyGraphResolver()
        resolver.add_functor(SimpleFunctor("a"))
        resolver.add_functor(
            DeclaredDependencyFunctor("consumer", {"required": ("a",), "lagged": ("b",)})
        )
        assert resolver.nodes["consumer"].dependencies == {"a"}
        assert resolver.nodes["consumer"].lagged_dependencies == {"b"}
        assert resolver.topological_sort() == ["a", "consumer"]

    def test_the_normalizer_has_a_single_owner(self):
        import relic.core.simulator.dependency_ops as dependency_ops
        import relic.core.simulator.scale_ops as scale_ops
        from relic.core.simulator import RELICSimulator

        from relic.core.dependency_graph import normalize_dependency_declaration as owner

        assert not hasattr(scale_ops, "_normalize_dependencies")
        # Modules may import the owner; none defines its own normalizer.
        for module in (scale_ops, dependency_ops):
            assert getattr(module, "normalize_dependency_declaration", owner) is owner
        assert not hasattr(dependency_ops, "split_dependencies")
        assert not hasattr(RELICSimulator, "_split_dependencies")
        assert not hasattr(DependencyGraphResolver, "_apply_structured_dependencies")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
