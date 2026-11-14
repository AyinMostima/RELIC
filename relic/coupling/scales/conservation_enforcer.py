"""
Conservation Enforcer Component for Scale Coupling

This module provides the conservation check used by scale transformations.
It audits source and target ledgers; it does not modify fields. A violated
ledger is reported (or rejected by the caller's fail-closed policy), never
corrected by rescaling.

Mathematical Expression:
    Conservation law: d/dt integralphi dV = integralS * n dA + integralQ dV
    Conservation error: epsilon = |integraltarget - integralsource| / |integralsource|
    
References:
    - Gross, L.J. et al. (2018). Conservation laws in ecosystem modeling
    - Ringler, T. et al. (2013). A unified approach to energy conservation
"""

import itertools
import math
from typing import Optional, Dict, Any, List, Tuple
import torch
from relic.core import device_policy
import torch.nn as nn
from torch import Tensor
import logging

from ..core.conservation_checker import (
    ConservationDiagnostics,
    check_energy_conservation,
    check_mass_conservation,
    require_relative_tolerance,
)

logger = logging.getLogger(__name__)


class ConservationEnforcer(nn.Module):
    """
    Conservation check for spatial/temporal scaling (audit only, no correction).

    Mathematical framework:
        Conservation error (relative), per batch b of the leading batch axes:
            epsilon_b = |int phi_target,b dOmega - int phi_source,b dOmega| / |int phi_source,b dOmega|
        Each batch ledger is conserved when its residual is within its accepted residual of the
        caller's tolerance, and the check is conserved only when every batch ledger is; the
        residuals of different batches are never added. Callers decide whether a violation is
        reported or rejected; the field itself is never rescaled here.

    The enforcer keeps, looks up and defaults no tolerance: every ``check_conservation``
    call passes the tolerance of its owner's parameter (for the ScaleManager facade the
    precision tolerance of the call's mode, ``ScaleManager.ledger_tolerance``).

    Parameters
    ----------
    conservation_properties : List[str]
        Names of properties to conserve (e.g., ["mass_carbon", "energy"])
    track_history : bool
        Whether to store diagnostics for post-run auditing

    References
    ----------
    - Gross, L.J. et al. (2018) Conservation laws in ecosystem modeling
    - Ringler, T. et al. (2013) Unified approach to energy conservation in ocean models
    - Jones, P.W. (1999) Conservative remapping schemes
    """

    def __init__(
        self,
        conservation_properties: List[str] = None,
        track_history: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize conservation enforcer.

        Parameters:
            conservation_properties : List[str]
                List of properties to conserve
            track_history : bool
                Whether to track conservation history
            device : Optional[torch.device]
                Computation device
        """
        super().__init__()

        self.conservation_properties = conservation_properties or ["mass"]
        self.track_history = track_history
        self.device = device or (
            device_policy.resolve_device(default="auto")
        )
        
        # Conservation history
        self.history: List[ConservationDiagnostics] = []

    @staticmethod
    def _apply_weights(
        data: Tensor,
        weights: Optional[Tensor] = None,
        volume: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply geometric weights prior to conservation diagnostics.

        References:
            - Jones (1999) conservative remapping schemes.
        Mathematical expression:
            weighted_phi = phi * w_area * w_volume
        Parameters:
            data: target field.
            weights: area weights (dimensionless).
            volume: control volumes (m^3 or m^2 depending on discretization).
        """
        result = data
        if weights is not None:
            weights = torch.as_tensor(weights, device=result.device, dtype=result.dtype)
            result = result * weights
        if volume is not None:
            volume = torch.as_tensor(volume, device=result.device, dtype=result.dtype)
            result = result * volume
        return result
        
    @staticmethod
    def _ledger_batch_shape(
        source_data: Tensor,
        target_data: Tensor,
        batch_ndim: Any,
        property_name: str,
    ) -> Tuple[int, ...]:
        """Shared leading batch shape of a source and a target field (one ledger per batch)."""
        if isinstance(batch_ndim, bool) or not isinstance(batch_ndim, int):
            raise TypeError(
                f"The ledger of '{property_name}' needs batch_ndim as an int (the number of leading "
                f"batch axes shared by source and target), got {batch_ndim!r}."
            )
        if batch_ndim < 0 or batch_ndim > min(source_data.dim(), target_data.dim()):
            raise ValueError(
                f"The ledger of '{property_name}' has batch_ndim={batch_ndim}; it must lie in "
                f"[0, {min(source_data.dim(), target_data.dim())}] for source shape "
                f"{tuple(source_data.shape)} and target shape {tuple(target_data.shape)}."
            )
        source_batch = tuple(source_data.shape[:batch_ndim])
        target_batch = tuple(target_data.shape[:batch_ndim])
        if source_batch != target_batch:
            raise ValueError(
                f"The ledger of '{property_name}' pairs source batch shape {source_batch} with target "
                f"batch shape {target_batch}; a ledger compares one source batch with the same "
                "target batch."
            )
        if math.prod(source_batch) == 0:
            raise ValueError(
                f"The ledger of '{property_name}' has the empty batch shape {source_batch}; there is "
                "no ledger to evaluate."
            )
        return source_batch

    @staticmethod
    def _require_weights_on_data(weighted: Tensor, data: Tensor, side: str, property_name: str) -> None:
        """Weights and volumes broadcast onto a field; they never widen its ledger."""
        if tuple(weighted.shape) != tuple(data.shape):
            raise ValueError(
                f"The {side} weights and volumes of '{property_name}' broadcast the field of shape "
                f"{tuple(data.shape)} to {tuple(weighted.shape)}; they must broadcast onto the field."
            )

    def _batch_ledger(
        self,
        property_name: str,
        source_terms: Tensor,
        target_terms: Tensor,
        rounding_dtype: Optional[torch.dtype],
        tolerance: float,
    ) -> Dict[str, Any]:
        """One ledger of one batch through the shared mass or energy checker."""
        if property_name.lower().find("energy") != -1:
            raw_result = check_energy_conservation(
                energy_in=source_terms,
                energy_out=target_terms,
                tolerance=tolerance,
                rounding_dtype=rounding_dtype,
            )
            total_dtype = torch.float64 if rounding_dtype is not None else source_terms.dtype
            source_total = float(source_terms.to(total_dtype).sum().item())
            target_total = float(target_terms.to(total_dtype).sum().item())
            residual = -float(raw_result["imbalance"])
            absolute_error = abs(float(raw_result["imbalance"]))
        else:
            raw_result = check_mass_conservation(
                source=source_terms,
                target=target_terms,
                tolerance=tolerance,
                variable_name=property_name,
                rounding_dtype=rounding_dtype,
            )
            source_total = float(raw_result["source_total"])
            target_total = float(raw_result["target_total"])
            residual = target_total - source_total
            absolute_error = float(raw_result["absolute_error"])
        return {
            "source_total": source_total,
            "target_total": target_total,
            "residual": residual,
            "absolute_error": absolute_error,
            "relative_error": float(raw_result["relative_error"]),
            "activity": float(raw_result["activity"]),
            "rounding_allowance": float(raw_result["rounding_allowance"]),
            "accepted_residual": float(raw_result["accepted_residual"]),
            "is_conserved": bool(raw_result["conserved"]),
        }

    def check_conservation(
        self,
        source_data: Tensor,
        target_data: Tensor,
        property_name: str = "mass",
        source_weights: Optional[Tensor] = None,
        target_weights: Optional[Tensor] = None,
        source_volume: Optional[Tensor] = None,
        target_volume: Optional[Tensor] = None,
        rounding_dtype: Optional[torch.dtype] = None,
        *,
        tolerance: float,
        batch_ndim: int,
    ) -> ConservationDiagnostics:
        """
        Check conservation between source and target data, one ledger per batch.

        Parameters:
            source_data : Tensor
                Source resolution data [*batch, *source_support]
            target_data : Tensor
                Target resolution data [*batch, *target_support]
            property_name : str
                Name of conserved property
            source_weights : Optional[Tensor]
                Weights for source data (e.g., cell areas); they broadcast onto source_data
            target_weights : Optional[Tensor]
                Weights for target data; they broadcast onto target_data
            source_volume : Optional[Tensor]
                Volume/area elements for source grid
            target_volume : Optional[Tensor]
                Volume/area elements for target grid
            rounding_dtype : Optional[torch.dtype]
                Coarsest floating dtype of the reduction that produced the target values
                (conservation_checker.coarsest_floating_dtype); the residual is then accepted
                up to the rounding bound of that dtype (conservation_checker.ledger_acceptance)
            tolerance : float
                Relative tolerance of this ledger (dimensionless, finite, in (0, 1)), read by
                the caller from its own parameter; required, never defaulted
            batch_ndim : int
                Number of leading axes that index independent ledgers (batches, for example time
                slices or ensemble members); source and target share these axes, and every other
                axis of a side is its support and is summed. 0 is one ledger over the whole
                fields. Required, never inferred.

        Returns:
            ConservationDiagnostics: ``is_conserved`` holds only when every batch ledger is
            accepted; ``absolute_error``, ``source_total``, ``target_total`` and the metadata
            ``activity``, ``rounding_allowance`` and ``accepted_residual`` are those of the batch
            with the largest residual (the first one on ties); ``relative_error`` is the largest
            batch relative error. The metadata also hold ``batch_shape``, ``batch_ledgers`` (one
            row per batch with its signed ``residual``) and ``failed_batches`` (the indices of the
            rejected batches). ``tolerance`` is the one used.
        """
        # References: relic/coupling/core/conservation_checker.py (check_mass_conservation,
        #   check_energy_conservation and the shared ledger_acceptance rule); Jones, P.W. (1999)
        #   Mon. Weather Rev. 127:2204-2210 (conservative ledgers); the row-wise acceptance of
        #   the covered-support ledger (covered_support_acceptance in regrid_conservative.py).
        # Mathematical expression: for every batch index b, L_src,b = sum_i phi_(b,i) w_i v_i and
        #   L_tgt,b = sum_j phi_(b,j) w_j v_j over the support axes only; e_b = |L_tgt,b - L_src,b|
        #   with signed residual r_b = L_tgt,b - L_src,b (energy: r_b = -(sum E_in,b - sum E_out,b));
        #   accepted_b iff e_b <= max(tol s_b, gamma_(2n+2)(u_dtype) a_b) with
        #   s_b = max(|L_src,b|, 1e-12) and a_b = sum_i |phi_(b,i) w_i v_i|; conserved iff every
        #   accepted_b; summary e = max_b e_b, epsilon = max_b e_b / s_b; the residuals of
        #   different batches are never added.
        # Parameter meanings: phi field values; w area weights; v volumes; b batch index over the
        #   batch_ndim leading axes shared by source and target; i, j support indices; tol the
        #   tolerance the caller passes for this ledger (ScaleManager: ledger_tolerance of the
        #   call's mode); n support cells of one source batch; gamma the Higham rounding bound
        #   of rounding_dtype (only when given).
        # Units: L, r, e, s and a in the ledger units of phi times w times v; tol, epsilon and
        #   gamma dimensionless.
        # Domain/constraints: audit only (no gradient, the field is never modified); tol finite in
        #   (0, 1) (require_relative_tolerance); batch_ndim an int in [0, min(ndim)] with equal
        #   leading shapes and at least one batch; weights and volumes broadcast onto their field
        #   and never widen it (ValueError otherwise).
        # Sign convention: r_b > 0 when the target holds more than the source (material added,
        #   for example by clipping up to a lower bound); e_b, s_b, a_b and epsilon non-negative.
        tolerance = require_relative_tolerance(
            tolerance, f"The ConservationEnforcer tolerance of '{property_name}'"
        )
        batch_shape = self._ledger_batch_shape(source_data, target_data, batch_ndim, property_name)
        with torch.no_grad():
            source_field = source_data.detach()
            target_field = target_data.detach()
            weighted_source = self._apply_weights(source_field, source_weights, source_volume)
            weighted_target = self._apply_weights(target_field, target_weights, target_volume)
            self._require_weights_on_data(weighted_source, source_field, "source", property_name)
            self._require_weights_on_data(weighted_target, target_field, "target", property_name)
            n_batches = math.prod(batch_shape)
            source_rows = weighted_source.reshape(n_batches, math.prod(source_field.shape[batch_ndim:]))
            target_rows = weighted_target.reshape(n_batches, math.prod(target_field.shape[batch_ndim:]))
            ledgers: List[Dict[str, Any]] = []
            for row, batch_index in enumerate(itertools.product(*(range(size) for size in batch_shape))):
                ledger = self._batch_ledger(
                    property_name, source_rows[row], target_rows[row], rounding_dtype, tolerance
                )
                ledgers.append({"batch_index": tuple(batch_index), **ledger})

        failed_batches = [ledger["batch_index"] for ledger in ledgers if not ledger["is_conserved"]]
        largest = max(ledgers, key=lambda ledger: ledger["absolute_error"])
        diagnostics = ConservationDiagnostics(
            property_name=property_name,
            source_total=largest["source_total"],
            target_total=largest["target_total"],
            absolute_error=largest["absolute_error"],
            relative_error=max(ledger["relative_error"] for ledger in ledgers),
            tolerance=tolerance,
            is_conserved=not failed_batches,
            metadata={
                'has_source_weights': source_weights is not None,
                'has_target_weights': target_weights is not None,
                'has_source_volume': source_volume is not None,
                'has_target_volume': target_volume is not None,
                'activity': largest['activity'],
                'rounding_allowance': largest['rounding_allowance'],
                'accepted_residual': largest['accepted_residual'],
                'batch_shape': batch_shape,
                'batch_ledgers': ledgers,
                'failed_batches': failed_batches,
            }
        )

        # Store in history if tracking
        if self.track_history:
            self.history.append(diagnostics)

        # Log warning if not conserved
        if failed_batches:
            worst = max(
                (ledger for ledger in ledgers if not ledger["is_conserved"]),
                key=lambda ledger: ledger["absolute_error"],
            )
            logger.warning(
                f"{property_name} conservation violated in {len(failed_batches)} of {len(ledgers)} "
                f"batch ledger(s) {failed_batches[:8]}: largest residual {worst['residual']:+.2e} "
                f"(relative {worst['relative_error']:.2e}) in batch {worst['batch_index']} exceeds its "
                f"accepted residual {worst['accepted_residual']:.2e} (tolerance {tolerance:.2e})"
            )

        return diagnostics
    
    def get_conservation_report(self) -> Dict[str, Any]:
        """
        Generate conservation report from history.
        
        Returns:
            Dict[str, Any]: Conservation statistics
        """
        if not self.history:
            return {
                'n_checks': 0,
                'n_violations': 0,
                'mean_error': 0.0,
                'max_error': 0.0,
                'properties': {}
            }
        
        # Aggregate by property
        property_stats = {}
        for prop in set(d.property_name for d in self.history):
            prop_history = [d for d in self.history if d.property_name == prop]
            errors = [d.relative_error for d in prop_history]
            violations = [d for d in prop_history if not d.is_conserved]
            
            property_stats[prop] = {
                'n_checks': len(prop_history),
                'n_violations': len(violations),
                'mean_error': sum(errors) / len(errors) if errors else 0.0,
                'max_error': max(errors) if errors else 0.0,
                'violation_rate': len(violations) / len(prop_history) if prop_history else 0.0
            }
        
        # Overall statistics
        all_errors = [d.relative_error for d in self.history]
        
        return {
            'n_checks': len(self.history),
            'n_violations': sum(1 for d in self.history if not d.is_conserved),
            'mean_error': sum(all_errors) / len(all_errors) if all_errors else 0.0,
            'max_error': max(all_errors) if all_errors else 0.0,
            'properties': property_stats
        }
    
    def reset_history(self) -> None:
        """Clear conservation history."""
        self.history.clear()
        logger.info("Conservation history cleared")


