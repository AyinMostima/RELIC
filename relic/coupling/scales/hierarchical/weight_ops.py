"""
Weight operations for hierarchical scaling.

The weighted mode of categorical fields (majority-area class) and the strict check of a
column-normalised hierarchy weight matrix (shape, finite and non-negative weights, member-column sums
against coupling.scales.defaults.hierarchy_numerical_tolerance).

Mathematical Expression:
    Weighted mode: mode_j = argmax_c sum_i w_ij [x_i = c]
    Column check: |sum_i W[i,j] - 1| <= tolerance
"""

import logging
import torch
from torch import Tensor

from relic.parameters import get_global_manager
from relic.coupling.core.conservation_checker import require_relative_tolerance
from relic.parameters.namespaces import ParameterNamespace

# Resolved namespaces to avoid repeated resolution
DEFAULTS_NS = ParameterNamespace.COUPLING_DEFAULTS.value

logger = logging.getLogger(__name__)


def _resolve_weight_tolerance() -> float:
    """Return coupling.scales.defaults.hierarchy_numerical_tolerance, the one tolerance of validate_weights."""
    pm = get_global_manager()
    tolerance = pm.get(
        "hierarchy_numerical_tolerance",
        namespace=DEFAULTS_NS,
        units="dimensionless",
        expected_type=float,
        required=True,
    )
    return require_relative_tolerance(tolerance, f"{DEFAULTS_NS}.hierarchy_numerical_tolerance")


class WeightOps:
    """
    Weight operations for hierarchical scaling (weighted mode and the strict weight-matrix check).
    """
    
    @staticmethod
    def weighted_mode(values: Tensor, weights: Tensor) -> Tensor:
        """Class with the largest weight share per target (the one weighted-mode law).

        Parameters:
            values : Tensor
                Class codes [n_source]
            weights : Tensor
                Non-negative weight of every source in every target [n_source, n_target]

        Returns:
            Tensor: Dominant class code per target [n_target]
        """
        # References: Friedl and Brodley (1997) IEEE Trans. Geosci. Remote Sens. 35:716-724
        #   (majority-area aggregation of land cover); CF Conventions 1.8 section 7.3 ("mode").
        # Mathematical expression: S_cj = sum_i w_ij [x_i = c];
        #   mode_j = min { c : S_cj = max_c' S_c'j } (ties go to the smallest class code).
        # Parameter meanings: x_i class code of source i; w_ij weight of source i in target j
        #   (area share A_i M_ij for hierarchies, the cell weight for a whole-field reduction);
        #   S_cj weight of class c in target j.
        # Units: w and S carry the weight units (for example m2); codes are dimensionless.
        # Domain/constraints: values one-dimensional; weights non-negative with one row per
        #   source; the count of cells of a class does not enter.
        # Sign convention: weights are non-negative; codes are returned unchanged.
        if values.dim() != 1 or weights.dim() != 2 or weights.shape[0] != values.shape[0]:
            raise ValueError(
                f"weighted_mode expects values [n] and weights [n, n_target]; got "
                f"{tuple(values.shape)} and {tuple(weights.shape)}."
            )
        if bool((weights < 0).any()):
            raise ValueError("weighted_mode needs non-negative weights.")
        classes, inverse = torch.unique(values, sorted=True, return_inverse=True)
        indicator = torch.nn.functional.one_hot(inverse, num_classes=classes.numel()).to(weights.dtype)
        class_weight = indicator.transpose(0, 1) @ weights
        # torch.argmax returns the first maximal index; classes are sorted ascending.
        return classes[torch.argmax(class_weight, dim=0)]

    @staticmethod
    def validate_weights(weights: Tensor, source_dim: int, target_dim: int) -> None:
        """
        Check a column-normalised membership matrix; every violation raises ValueError.

        Parameters:
            weights : Tensor
                Membership weights W[i, j] of source i in target j, each target column
                normalised over its members
            source_dim : int
                Expected number of source units
            target_dim : int
                Expected number of target units
        """
        # References: Jones (1999) Mon. Weather Rev. 127:2204-2210 (normalised remapping weights).
        # Mathematical expression: W has shape [n_source, n_target], every W_ij is finite and >= 0,
        #   and |sum_i W_ij - 1| <= tol for every target j with members (sum_i W_ij > 0).
        # Parameter meanings: W_ij membership weight of source i in target j; tol =
        #   coupling.scales.defaults.hierarchy_numerical_tolerance.
        # Units: W and tol [1].
        # Domain/constraints: a target without members (a zero column) passes; it receives no
        #   member value, as the aggregation kernels define; any other violation raises.
        # Sign convention: weights are non-negative.
        tolerance = _resolve_weight_tolerance()
        if tuple(weights.shape) != (source_dim, target_dim):
            raise ValueError(
                f"Weight matrix shape {tuple(weights.shape)} differs from ({source_dim}, {target_dim})."
            )
        values = weights.to_dense() if weights.is_sparse else weights
        values = values.detach().to(dtype=torch.float64)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("Weight matrix contains non-finite values.")
        if bool((values < 0).any()):
            raise ValueError("Weight matrix contains negative weights.")
        column_sums = values.sum(dim=0)
        members = column_sums > 0
        deviation = (column_sums[members] - 1.0).abs()
        if deviation.numel() > 0 and float(deviation.max().item()) > tolerance:
            worst = int(torch.nonzero(members).reshape(-1)[int(torch.argmax(deviation).item())].item())
            raise ValueError(
                f"Weight column {worst} sums to {float(column_sums[worst].item()):.12g}; member columns must "
                f"sum to 1 within coupling.scales.defaults.hierarchy_numerical_tolerance = {tolerance}."
            )
