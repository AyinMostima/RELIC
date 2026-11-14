"""
Graph topology algebra for axiomatic self-inspection.

Provides pure-function utilities to extract the incidence matrix,
compute its left null space (conservation laws), rank, and cycle
rank (Betti number beta_1) from a GraphIR instance.

References
----------
- Godsil & Royle (2001), Algebraic Graph Theory, Ch. 8:
  Incidence matrix B encodes node-link adjacency.  Its left null
  space ker(B^T) yields conservation laws; cycle rank
  beta_1 = m - n + c gives gauge degrees of freedom.
- Strang (1988), Linear Algebra and Its Applications:
  The four fundamental subspaces of a matrix.  Left null space of
  B = vectors in R^n orthogonal to column space of B.

Mathematical expressions
------------------------
- B[i, j] = +1  if link j enters node i,
             -1  if link j leaves node i,
              0  otherwise.
- left null space = {x : B^T x = 0} computed via SVD.
- rank(B) = number of non-zero singular values above tolerance.
- nullity = n - rank(B)  (dimension of left null space).
- beta_1 = m - rank(B)   (cycle rank / first Betti number),
  where n = number of nodes, m = number of links.

Parameter meanings
------------------
- graph_ir : GraphIR
    The graph intermediate representation to analyze.
- tol : float
    Singular value threshold for distinguishing zero from non-zero.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .graph_ir import GraphIR

logger = logging.getLogger(__name__)


def _svd_full(B: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute full SVD with a stable backend on Windows.

    Uses torch.linalg.svd to avoid intermittent process aborts observed
    with numpy.linalg.svd in this environment, while preserving the
    original algebraic behavior and return types.
    """
    B_tensor = torch.as_tensor(B, dtype=torch.float64)
    U, sigma, Vh = torch.linalg.svd(B_tensor, full_matrices=True)
    return (
        U.detach().cpu().numpy(),
        sigma.detach().cpu().numpy(),
        Vh.detach().cpu().numpy(),
    )


def build_incidence_matrix(graph_ir: GraphIR) -> Tuple[np.ndarray, List[str], List[int]]:
    """Build the node-link incidence matrix B from a GraphIR.

    References
    ----------
    Godsil & Royle (2001), Algebraic Graph Theory, Definition 8.1.1.

    Mathematical expression
    -----------------------
    B is an n x m matrix where n = |nodes|, m = |links|.
    B[i, j] = -1 if link j leaves node i (source),
              +1 if link j enters node i (target),
               0 otherwise.

    Parameters
    ----------
    graph_ir : GraphIR
        Graph to analyze.  Must have explicit links (call
        ``_infer_links_from_inputs`` first if needed).

    Returns
    -------
    B : np.ndarray, shape (n, m)
        Signed incidence matrix.
    node_ids : List[str]
        Row-label ordering of nodes.
    link_indices : List[int]
        Column indices (0..m-1) for reference.
    """
    node_ids = sorted(graph_ir.nodes.keys())
    node_index = {nid: idx for idx, nid in enumerate(node_ids)}
    links = graph_ir.links or graph_ir._infer_links_from_inputs()
    n = len(node_ids)
    m = len(links)

    B = np.zeros((n, m), dtype=np.float64)
    for j, link in enumerate(links):
        src_idx = node_index.get(link.source.node)
        dst_idx = node_index.get(link.target.node)
        if src_idx is not None:
            B[src_idx, j] = -1.0
        if dst_idx is not None:
            B[dst_idx, j] = +1.0

    return B, node_ids, list(range(m))


def compute_left_null_space(
    B: np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Compute the left null space of incidence matrix B via SVD.

    The left null space ker(B^T) contains vectors c such that
    B^T c = 0.  Each such vector represents a conservation law:
    the weighted sum of node stocks c^T x is invariant under any
    flux consistent with B.

    References
    ----------
    Strang (1988), Section 3.1: Four fundamental subspaces.

    Mathematical expression
    -----------------------
    B = U Sigma V^T
    Left null space = rows of U^T corresponding to zero singular
    values, i.e. U[:, rank:].

    Parameters
    ----------
    B : np.ndarray, shape (n, m)
        Incidence matrix.
    tol : float
        Threshold below which a singular value is treated as zero.

    Returns
    -------
    null_space : np.ndarray, shape (d, n)
        Each row is a conservation law vector.  d = n - rank(B).
    """
    n = B.shape[0] if B.ndim == 2 else 0
    if B.size == 0:
        # For B with zero columns, B^T x = 0 holds for all x in R^n.
        # Return a full basis of the left null space as identity rows.
        return np.eye(n, dtype=np.float64)

    U, sigma, _Vt = _svd_full(B)
    rank = int(np.sum(sigma > tol))
    # Left null space = columns of U beyond the rank
    null_vectors = U[:, rank:].T  # shape (d, n)
    return null_vectors


def compute_rank(B: np.ndarray, tol: float = 1e-10) -> int:
    """Compute the numerical rank of incidence matrix B.

    Parameters
    ----------
    B : np.ndarray
        Incidence matrix.
    tol : float
        Singular value threshold.

    Returns
    -------
    int
        Numerical rank.
    """
    if B.size == 0:
        return 0
    _U, sigma, _Vh = _svd_full(B)
    return int(np.sum(sigma > tol))


def compute_cycle_rank(B: np.ndarray, tol: float = 1e-10) -> int:
    """Compute the first Betti number (cycle rank) of the graph.

    References
    ----------
    Godsil & Royle (2001), Corollary 8.3.3:
    beta_1 = m - rank(B) for a connected graph; equivalently
    beta_1 = m - n + c where c is the number of connected
    components.

    Mathematical expression
    -----------------------
    beta_1 = m - rank(B)
    This counts the number of independent cycles, i.e. the
    dimension of the cycle space (gauge degrees of freedom).

    Parameters
    ----------
    B : np.ndarray, shape (n, m)
        Incidence matrix.
    tol : float
        Singular value threshold.

    Returns
    -------
    int
        Cycle rank (first Betti number).
    """
    m = B.shape[1] if B.ndim == 2 else 0
    return max(0, m - compute_rank(B, tol))


def get_topology_signature(
    graph_ir: GraphIR,
    tol: float = 1e-10,
) -> Dict[str, Any]:
    """Compute the full topology signature for a GraphIR.

    Returns a dictionary containing all algebraic invariants
    needed for the Structural Ecology axiomatic verification:
    rank, nullity, Betti number, and conservation law vectors.

    Parameters
    ----------
    graph_ir : GraphIR
        Graph to analyze.
    tol : float
        SVD tolerance.

    Returns
    -------
    Dict[str, Any]
        Keys: n_nodes, n_links, rank, nullity, betti_number,
        conservation_vectors, node_ids.
    """
    B, node_ids, _ = build_incidence_matrix(graph_ir)
    n = len(node_ids)
    m = B.shape[1] if B.ndim == 2 and B.size > 0 else 0
    rank = compute_rank(B, tol)
    nullity = n - rank
    betti = max(0, m - rank)
    null_space = compute_left_null_space(B, tol)

    return {
        "n_nodes": n,
        "n_links": m,
        "rank": rank,
        "nullity": nullity,
        "betti_number": betti,
        "conservation_vectors": null_space.tolist(),
        "node_ids": node_ids,
    }
