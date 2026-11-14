"""Unit tests for topology_algebra module (Patch P1).

Verifies incidence matrix construction, SVD-based left null space
(conservation laws), rank, and Betti number (cycle rank) computation
from GraphIR instances.

References:
- Godsil & Royle (2001), Algebraic Graph Theory, Ch. 8
- Strang (1988), Linear Algebra and Its Applications
"""

import numpy as np

from relic.orchestration.graph_ir import GraphIR, LinkEndpoint, LinkSpec, NodeSpec
from relic.orchestration.topology_algebra import (
    build_incidence_matrix,
    compute_cycle_rank,
    compute_left_null_space,
    compute_rank,
    get_topology_signature,
)


def _make_chain_graph() -> GraphIR:
    """A -> B -> C  (3 nodes, 2 links, no cycle)."""
    return GraphIR(
        schema_version="1.0",
        nodes={
            "A": NodeSpec(class_type="source"),
            "B": NodeSpec(class_type="process"),
            "C": NodeSpec(class_type="sink"),
        },
        links=[
            LinkSpec(
                source=LinkEndpoint(node="A", port="out"),
                target=LinkEndpoint(node="B", port="in"),
            ),
            LinkSpec(
                source=LinkEndpoint(node="B", port="out"),
                target=LinkEndpoint(node="C", port="in"),
            ),
        ],
    )


def _make_cycle_graph() -> GraphIR:
    """A -> B -> C -> A  (3 nodes, 3 links, 1 cycle)."""
    return GraphIR(
        schema_version="1.0",
        nodes={
            "A": NodeSpec(class_type="process"),
            "B": NodeSpec(class_type="process"),
            "C": NodeSpec(class_type="process"),
        },
        links=[
            LinkSpec(
                source=LinkEndpoint(node="A", port="out"),
                target=LinkEndpoint(node="B", port="in"),
            ),
            LinkSpec(
                source=LinkEndpoint(node="B", port="out"),
                target=LinkEndpoint(node="C", port="in"),
            ),
            LinkSpec(
                source=LinkEndpoint(node="C", port="out"),
                target=LinkEndpoint(node="A", port="in"),
            ),
        ],
    )


def _make_diamond_graph() -> GraphIR:
    """A -> B, A -> C, B -> D, C -> D  (4 nodes, 4 links, 1 cycle)."""
    return GraphIR(
        schema_version="1.0",
        nodes={
            "A": NodeSpec(class_type="source"),
            "B": NodeSpec(class_type="process"),
            "C": NodeSpec(class_type="process"),
            "D": NodeSpec(class_type="sink"),
        },
        links=[
            LinkSpec(source=LinkEndpoint(node="A", port="out"), target=LinkEndpoint(node="B", port="in")),
            LinkSpec(source=LinkEndpoint(node="A", port="out2"), target=LinkEndpoint(node="C", port="in")),
            LinkSpec(source=LinkEndpoint(node="B", port="out"), target=LinkEndpoint(node="D", port="in1")),
            LinkSpec(source=LinkEndpoint(node="C", port="out"), target=LinkEndpoint(node="D", port="in2")),
        ],
    )


def _make_single_node_graph() -> GraphIR:
    """Single node, no links."""
    return GraphIR(
        schema_version="1.0",
        nodes={"solo": NodeSpec(class_type="standalone")},
        links=[],
    )


class TestIncidenceMatrix:
    """Tests for build_incidence_matrix."""

    def test_chain_shape(self) -> None:
        graph = _make_chain_graph()
        B, node_ids, link_indices = build_incidence_matrix(graph)
        assert B.shape == (3, 2), f"Expected (3,2), got {B.shape}"
        assert len(node_ids) == 3
        assert len(link_indices) == 2

    def test_chain_values(self) -> None:
        graph = _make_chain_graph()
        B, node_ids, _ = build_incidence_matrix(graph)
        # Nodes sorted: A=0, B=1, C=2
        # Link 0: A->B => B[0,0]=-1, B[1,0]=+1
        # Link 1: B->C => B[1,1]=-1, B[2,1]=+1
        assert node_ids == ["A", "B", "C"]
        assert B[0, 0] == -1.0
        assert B[1, 0] == +1.0
        assert B[1, 1] == -1.0
        assert B[2, 1] == +1.0
        # All other entries zero
        assert B[0, 1] == 0.0
        assert B[2, 0] == 0.0

    def test_cycle_shape(self) -> None:
        graph = _make_cycle_graph()
        B, node_ids, _ = build_incidence_matrix(graph)
        assert B.shape == (3, 3)

    def test_single_node(self) -> None:
        graph = _make_single_node_graph()
        B, node_ids, _ = build_incidence_matrix(graph)
        assert B.shape == (1, 0)
        assert node_ids == ["solo"]

    def test_column_sum_zero(self) -> None:
        """Each column of B sums to zero (one +1 and one -1 per link)."""
        graph = _make_diamond_graph()
        B, _, _ = build_incidence_matrix(graph)
        col_sums = B.sum(axis=0)
        np.testing.assert_allclose(col_sums, 0.0, atol=1e-15)


class TestLeftNullSpace:
    """Tests for compute_left_null_space."""

    def test_chain_has_one_conservation_law(self) -> None:
        """A chain of 3 nodes with 2 links: rank=2, nullity=1."""
        graph = _make_chain_graph()
        B, _, _ = build_incidence_matrix(graph)
        null_space = compute_left_null_space(B)
        # nullity = n - rank = 3 - 2 = 1
        assert null_space.shape[0] == 1, f"Expected 1 conservation law, got {null_space.shape[0]}"

    def test_null_space_orthogonality(self) -> None:
        """Verify B^T c = 0 for each conservation vector c."""
        graph = _make_chain_graph()
        B, _, _ = build_incidence_matrix(graph)
        null_space = compute_left_null_space(B)
        for c in null_space:
            residual = B.T @ c
            np.testing.assert_allclose(residual, 0.0, atol=1e-10)

    def test_cycle_conservation(self) -> None:
        """3-node cycle: rank=2, nullity=1."""
        graph = _make_cycle_graph()
        B, _, _ = build_incidence_matrix(graph)
        null_space = compute_left_null_space(B)
        assert null_space.shape[0] == 1

    def test_single_node_trivial(self) -> None:
        """Single node with no links: entire space is null space."""
        graph = _make_single_node_graph()
        B, _, _ = build_incidence_matrix(graph)
        null_space = compute_left_null_space(B)
        assert null_space.shape == (1, 1)

    def test_diamond_null_space(self) -> None:
        """Diamond graph: rank=3, nullity=1."""
        graph = _make_diamond_graph()
        B, _, _ = build_incidence_matrix(graph)
        null_space = compute_left_null_space(B)
        assert null_space.shape[0] == 1
        for c in null_space:
            residual = B.T @ c
            np.testing.assert_allclose(residual, 0.0, atol=1e-10)


class TestBettiNumber:
    """Tests for compute_cycle_rank (first Betti number)."""

    def test_tree_betti_zero(self) -> None:
        """Chain (tree) graph: beta_1 = 0."""
        graph = _make_chain_graph()
        B, _, _ = build_incidence_matrix(graph)
        assert compute_cycle_rank(B) == 0

    def test_cycle_betti_one(self) -> None:
        """3-node cycle: beta_1 = 1."""
        graph = _make_cycle_graph()
        B, _, _ = build_incidence_matrix(graph)
        assert compute_cycle_rank(B) == 1

    def test_diamond_betti_one(self) -> None:
        """Diamond graph has 1 independent cycle: beta_1 = 1."""
        graph = _make_diamond_graph()
        B, _, _ = build_incidence_matrix(graph)
        assert compute_cycle_rank(B) == 1

    def test_single_node_betti_zero(self) -> None:
        graph = _make_single_node_graph()
        B, _, _ = build_incidence_matrix(graph)
        assert compute_cycle_rank(B) == 0


class TestRank:
    """Tests for compute_rank."""

    def test_chain_rank(self) -> None:
        graph = _make_chain_graph()
        B, _, _ = build_incidence_matrix(graph)
        assert compute_rank(B) == 2

    def test_cycle_rank(self) -> None:
        graph = _make_cycle_graph()
        B, _, _ = build_incidence_matrix(graph)
        assert compute_rank(B) == 2

    def test_diamond_rank(self) -> None:
        graph = _make_diamond_graph()
        B, _, _ = build_incidence_matrix(graph)
        assert compute_rank(B) == 3


class TestTopologySignature:
    """Tests for get_topology_signature (integration)."""

    def test_chain_signature(self) -> None:
        graph = _make_chain_graph()
        sig = get_topology_signature(graph)
        assert sig["n_nodes"] == 3
        assert sig["n_links"] == 2
        assert sig["rank"] == 2
        assert sig["nullity"] == 1
        assert sig["betti_number"] == 0
        assert len(sig["conservation_vectors"]) == 1
        assert sig["node_ids"] == ["A", "B", "C"]

    def test_cycle_signature(self) -> None:
        graph = _make_cycle_graph()
        sig = get_topology_signature(graph)
        assert sig["n_nodes"] == 3
        assert sig["n_links"] == 3
        assert sig["rank"] == 2
        assert sig["nullity"] == 1
        assert sig["betti_number"] == 1

    def test_returns_all_fields(self) -> None:
        graph = _make_chain_graph()
        sig = get_topology_signature(graph)
        required_keys = {
            "n_nodes", "n_links", "rank", "nullity",
            "betti_number", "conservation_vectors", "node_ids",
        }
        assert required_keys.issubset(set(sig.keys()))
