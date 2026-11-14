from relic.core.dependency_graph import DependencyGraphResolver


class _GraphFunctor:
    """Minimal declaration object for graph-order tests."""

    def __init__(self, name: str, deps=None):
        self.name = name
        self._deps = deps or []

    def get_dependencies(self):
        return list(self._deps)


def test_dependency_graph_orders_but_does_not_execute_state() -> None:
    """State application belongs to Simulator and StateManager, not this graph."""
    resolver = DependencyGraphResolver()
    resolver.add_functor(_GraphFunctor("provider"))
    resolver.add_functor(_GraphFunctor("consumer", deps=["provider"]))

    assert resolver.topological_sort() == ["provider", "consumer"]
    assert not hasattr(resolver, "execute_in_order")


def _build_resolver(use_networkx: bool, names, edges):
    """Register ``names`` in order; ``edges`` maps consumer -> providers."""
    resolver = DependencyGraphResolver(use_networkx=use_networkx)
    for name in names:
        resolver.add_functor(_GraphFunctor(name, deps=edges.get(name, [])))
    return resolver


def test_topological_tie_break_is_registration_order_on_both_paths() -> None:
    """Kahn and networkx must agree, and free choices follow the registration total order (D84)."""
    edges = {"d": ["a"], "e": ["b", "c"]}
    forward = ["c", "a", "b", "d", "e"]
    reverse = ["b", "a", "c", "e", "d"]

    orders_forward = {
        use_nx: _build_resolver(use_nx, forward, edges).topological_sort() for use_nx in (True, False)
    }
    orders_reverse = {
        use_nx: _build_resolver(use_nx, reverse, edges).topological_sort() for use_nx in (True, False)
    }
    assert orders_forward[True] == orders_forward[False] == ["c", "a", "b", "d", "e"]
    assert orders_reverse[True] == orders_reverse[False] == ["b", "a", "c", "e", "d"]

    # Repeated sorts of the same graph are identical (stable).
    resolver = _build_resolver(False, forward, edges)
    assert resolver.topological_sort() == resolver.topological_sort(handle_optional=False)


def test_topological_tie_break_survives_snapshot_restore() -> None:
    resolver = _build_resolver(False, ["z", "y", "x"], {})
    snapshot = resolver.capture_mutation_state()
    resolver.add_functor(_GraphFunctor("w"))
    assert resolver.topological_sort() == ["z", "y", "x", "w"]
    resolver.restore_mutation_state(snapshot)
    assert resolver.topological_sort() == ["z", "y", "x"]
