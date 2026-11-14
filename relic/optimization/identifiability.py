"""
Jacobian-based identifiability diagnostics for observation operators.

This module computes rank/nullity/condition diagnostics from the SVD of
the observation Jacobian matrix to detect unidentifiable state-space
directions in data assimilation workflows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def _to_float(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("Jacobian entries must be scalar tensors.")
        return float(value.detach().cpu().item())
    return float(value)


def _jacobian_dict_to_matrix(
    jacobian: dict[str, dict[str, Tensor | float | int]],
) -> tuple[np.ndarray, list[str], list[str]]:
    obs_vars = sorted(jacobian.keys())
    state_vars_set = set()
    for obs_var in obs_vars:
        state_vars_set.update(jacobian.get(obs_var, {}).keys())
    state_vars = sorted(state_vars_set)

    if not obs_vars or not state_vars:
        return (
            np.zeros((len(obs_vars), len(state_vars)), dtype=np.float64),
            obs_vars,
            state_vars,
        )

    # Every row must list every state variable (an explicit 0.0 for no
    # sensitivity). A missing entry is an incomplete Jacobian; reading it as
    # zero would add a false null-space direction.
    missing = [
        f"{obs_var}/{state_var}"
        for obs_var in obs_vars
        for state_var in state_vars
        if state_var not in jacobian[obs_var]
    ]
    if missing:
        raise KeyError(f"Jacobian entries missing (obs/state): {', '.join(missing)}")

    matrix = np.zeros((len(obs_vars), len(state_vars)), dtype=np.float64)
    for i, obs_var in enumerate(obs_vars):
        row = jacobian[obs_var]
        for j, state_var in enumerate(state_vars):
            matrix[i, j] = _to_float(row[state_var])
    return matrix, obs_vars, state_vars


def _to_matrix(
    jacobian: dict[str, dict[str, Tensor | float | int]] | Tensor | np.ndarray,
) -> tuple[np.ndarray, dict[str, list[str]]]:
    if isinstance(jacobian, dict):
        matrix, obs_vars, state_vars = _jacobian_dict_to_matrix(jacobian)
        return matrix, {"obs_vars": obs_vars, "state_vars": state_vars}
    if isinstance(jacobian, Tensor):
        return (
            jacobian.detach().cpu().to(dtype=torch.float64).numpy(),
            {"obs_vars": [], "state_vars": []},
        )
    if isinstance(jacobian, np.ndarray):
        return jacobian.astype(np.float64, copy=False), {"obs_vars": [], "state_vars": []}
    raise TypeError("jacobian must be nested dict, torch.Tensor, or numpy.ndarray.")


def compute_identifiability(
    jacobian: dict[str, dict[str, Tensor | float | int]] | Tensor | np.ndarray,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """
    SVD of the observation Jacobian H to detect unidentifiable directions.

    References:
    - Cressie and Wikle (2011), Statistics for Spatio-Temporal Data
    - Golub and Van Loan (2013), Matrix Computations, 4th ed.
    Mathematical expression:
    - H = U * Sigma * V^T
    - tau = max(tol, max(m, n) * eps) * sigma_max
    - rank(H) = |{sigma_i: sigma_i > tau}|
    - nullity(H) = n - rank(H)
    - null(H) = span{V[:, rank(H):n]}
    Parameters:
    - jacobian: nested dict {obs_var: {state_var: Tensor}} or matrix Jacobian
    - tol: finite, non-negative relative singular-value tolerance
    """
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and non-negative.")

    matrix, labels = _to_matrix(jacobian)
    if matrix.ndim != 2:
        raise ValueError("jacobian must be a two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("jacobian entries must be finite.")

    n_obs, n_state = matrix.shape

    if n_obs == 0 or n_state == 0:
        return {
            "rank": 0,
            "nullity": n_state,
            "condition_number": float("inf"),
            "singular_values": [],
            "unidentifiable_dirs": np.eye(n_state, dtype=np.float64).tolist(),
            "rank_tolerance": 0.0,
            "relative_tolerance": float(tol),
            "shape": [n_obs, n_state],
            **labels,
        }

    matrix_tensor = torch.as_tensor(matrix, dtype=torch.float64)

    # References:
    # - Cressie and Wikle (2011), ch. on inverse problems in DA
    # Mathematical expression:
    # - H = U * Sigma * V^T
    # - tau = max(tol, max(m, n) * eps) * sigma_max
    # - rank(H) = |{sigma_i: sigma_i > tau}|
    # - nullity(H) = n_state - rank(H)
    # Parameter meanings:
    # - H: observation Jacobian matrix
    # - tol: user-declared relative singular-value tolerance
    # - eps: float64 machine precision
    # - sigma_max: largest singular value of H
    _, sigma, right_vectors_transpose = torch.linalg.svd(
        matrix_tensor,
        full_matrices=True,
    )

    sigma_np = sigma.detach().cpu().numpy()
    sigma_max = float(sigma_np[0]) if sigma_np.size else 0.0
    numerical_floor = max(n_obs, n_state) * np.finfo(np.float64).eps
    rank_tolerance = float(max(float(tol), numerical_floor) * sigma_max)
    rank = int(np.sum(sigma_np > rank_tolerance))
    nullity = int(n_state - rank)

    if rank < n_state:
        condition_number = float("inf")
    else:
        sigma_min = float(sigma_np[n_state - 1])
        condition_number = float(sigma_max / sigma_min)

    right_vectors_transpose_np = right_vectors_transpose.detach().cpu().numpy()
    unidentifiable_dirs = right_vectors_transpose_np[rank:, :]

    return {
        "rank": rank,
        "nullity": nullity,
        "condition_number": condition_number,
        "singular_values": sigma_np.tolist(),
        "unidentifiable_dirs": unidentifiable_dirs.tolist(),
        "rank_tolerance": rank_tolerance,
        "relative_tolerance": float(tol),
        "shape": [n_obs, n_state],
        **labels,
    }


def get_identifiability_summary(
    jacobian: dict[str, dict[str, Tensor | float | int]] | Tensor | np.ndarray,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Return summary dict with rank, nullity, condition_number, singular_values."""
    details = compute_identifiability(jacobian, tol=tol)
    return {
        "rank": details["rank"],
        "nullity": details["nullity"],
        "condition_number": details["condition_number"],
        "singular_values": details["singular_values"],
        "rank_tolerance": details["rank_tolerance"],
        "relative_tolerance": details["relative_tolerance"],
        "shape": details["shape"],
    }
