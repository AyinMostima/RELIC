"""
Spatial coupling adapters built on the canonical scale utilities.

Temporal aggregation is owned only by
`relic.coupling.scales.temporal.aggregator.TemporalAggregator`. This module
retains spatial and conservative adapters and does not expose a second temporal
Functor, alias, or functional reduction path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

import torch
from relic.core import device_policy
from torch import Tensor

from relic.core.abstractions import SpatialScalingBase
from relic.core.functor import ScaleCouplingFunctor, FunctorMetadata
from relic.core.functor import ProcessCategory, ExecutionPhase, SpatialScale, VariableKind
from relic.coupling.core.policy import ScalingMode
from relic.coupling.scales.types import (
    ScalingStrategy,
    ScalingAudit,
    SpatialQuantityBasis,
    resolve_spatial_basis,
)
from relic.coupling.scales.alias_utils import (
    aggregate_spatial_alias,
    remap_conservative_alias,
)
from relic.coupling.scales.variable_catalog import find_catalog_entry
from relic.parameters.namespaces import ParameterNamespace

# Resolved namespaces to avoid repeated resolution and enum/string mixing
TEMPORAL_NS = ParameterNamespace.COUPLING_TEMPORAL.value
SPATIAL_NS = ParameterNamespace.COUPLING_SPATIAL.value
PHYSICAL_NS = ParameterNamespace.COUPLING_SCALES_PHYSICAL.value
DEFAULT_NS = ParameterNamespace.COUPLING_SCALES_DEFAULTS.value
# ParameterNamespace currently does not expose efficiencies; keep a local constant to avoid hardcoded duplicates.
EFFICIENCIES_NS = "coupling.scales.efficiencies"
logger = logging.getLogger(__name__)

__all__ = [
    "SpatialAggregation",
    "ConservativeRemap",
    "SpatialScaleCoupling",
    "GridRemapping",
    "ConservativeGridRemapping",
    "aggregate_spatial",
    "conservative_remap",
]


def _lookup_variable_metadata(
    name: str,
    parameter_accessor,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return ``(metadata, source)`` from the call overrides or the YAML variable catalogs.

    ``source`` is ``"call"`` for ``variable_metadata`` entries, ``"parameter:<namespace>"``
    for catalog entries (matched by the shared variable_catalog.find_catalog_entry) and
    ``"missing"`` when no entry exists. Names are not rewritten (no suffix stripping).
    """
    if overrides:
        for key in (name, name.lower()):
            if key in overrides:
                meta = overrides[key]
                if not isinstance(meta, dict):
                    raise TypeError(
                        f"variable_metadata[{key!r}] must be a mapping, got {type(meta).__name__}."
                    )
                return meta, "call"

    def _read_catalog(namespace: str) -> Optional[Dict[str, Any]]:
        if hasattr(parameter_accessor, "get_parameter_value"):
            return parameter_accessor.get_parameter_value(
                "variables",
                namespace=namespace,
                expected_type=dict,
                validate=False,
                required=False,
            )
        return parameter_accessor.get(
            "variables",
            namespace=namespace,
            expected_type=dict,
            required=False,
        )

    found = find_catalog_entry(
        name,
        (SPATIAL_NS, TEMPORAL_NS, PHYSICAL_NS, DEFAULT_NS, EFFICIENCIES_NS),
        _read_catalog,
    )
    if found is None:
        return None, "missing"
    return found[0], f"parameter:{found[1]}"


def _resolve_variable_basis(
    name: str,
    parameter_accessor,
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[SpatialQuantityBasis, str]:
    """Resolve the explicit spatial basis of one variable; no name heuristic exists.

    The metadata entry comes from the call's ``variable_metadata`` or from the YAML
    ``coupling.scales.*.variables`` catalogs and is interpreted by the shared
    ``resolve_spatial_basis``. An undeclared variable, or one declared only as
    ``extensive``, raises.
    """
    meta, source = _lookup_variable_metadata(name, parameter_accessor, overrides)
    if meta is None:
        raise ValueError(
            f"Variable {name!r} has no metadata in variable_metadata or the "
            "coupling.scales.*.variables catalogs; declare spatial_basis (intensive, "
            "cell_total, area_density or categorical) for spatial operators."
        )
    return resolve_spatial_basis(name, meta), source


# SpatialAggregation selectors (coupling.scales.spatial.aggregation_strategy). "area_weighted"
# applies the whole-support law of each variable's declared basis; every other selector names one
# basis law (aggregate_spatial_alias raises for a variable of another basis). No selector defines
# a maximum, minimum or median.
_AGGREGATION_SELECTORS: Dict[str, Optional[ScalingStrategy]] = {
    "area_weighted": None,
    "mean": ScalingStrategy.AREA_WEIGHTED_MEAN,
    "area_weighted_mean": ScalingStrategy.AREA_WEIGHTED_MEAN,
    "sum": ScalingStrategy.AREA_WEIGHTED_SUM,
    "total": ScalingStrategy.AREA_WEIGHTED_SUM,
    "area_weighted_sum": ScalingStrategy.AREA_WEIGHTED_SUM,
    "conservative": ScalingStrategy.CONSERVATIVE,
}


class SpatialAggregation(SpatialScalingBase, ScaleCouplingFunctor):
    """
    Whole-support spatial aggregation through the shared ``aggregate_spatial_alias`` law.

    Reduces each requested field over its whole support to one value. The law is selected
    by the variable's explicit SpatialQuantityBasis (call ``variable_metadata`` or the YAML
    variable catalogs), never by its name, and the value keeps that basis.

    Parameters
    ----------
    name : str, optional
        Functor identifier.
    aggregation_strategy : str, optional
        Selector (dimensionless): ``area_weighted`` applies each variable's basis law;
        ``mean``/``area_weighted_mean`` name the intensive and area-density law,
        ``sum``/``total``/``area_weighted_sum`` the cell-total law and ``conservative``
        the conservative law of every basis except categorical. A selector that conflicts
        with a variable's basis raises; ``max``, ``min`` and ``median`` are rejected.
    preserve_totals : bool, optional
        Strict ledger (dimensionless): the residual of an amount variable above the
        ``coupling.scales.defaults.conservation_tolerance`` tolerance raises; otherwise it is
        reported. It never turns an intensive mean into a sum.
    weighting_type : str, optional
        Measure of the weights (dimensionless selector); only ``area`` exists: the weights are
        cell areas (or another area measure), the declared basis selects the law, and an area
        density conserves its amount only with area weights.
    device : torch.device, optional
        Target computation device.
    source_scale : str, optional
        Source grid alias.
    target_scale : str, optional
        Target grid alias.
    config : dict, optional
        Extra FunctorBase configuration.
    parameter_manager : ParameterManager, optional
        Parameter manager instance; defaults to the global registry.

    Returns
    -------
    Dict[str, Tensor]
        Aggregated outputs keyed as ``{var}_aggregated`` in the input basis; the call's
        ``ScalingAudit`` is ``last_scaling_audit`` (``quantity_basis``, ``output_basis``,
        ``reduction``, ``conservation_audit`` and ``conservation_audited_variables``).

    References
    ----------
    CF-1.8 area-weighted spatial aggregation guidance
    Jones, P.W. (1999). Conservative remapping schemes.

    Mathematical Expressions
    ------------------------
    intensive and area_density: ``phi = sum_i x_i w_i / sum_i w_i`` with ``sum_i w_i > 0``
    (``w_i`` default 1 for intensive fields; an area density needs its cell areas ``A_i``).
    cell_total: ``Q = sum_i X_i`` (weights are not applied).
    categorical: ``c* = argmax_c sum_{i: x_i = c} w_i`` (ties to the smallest code).
    Ledger of an amount: ``sum_i x_i A_i`` against ``phi sum_i A_i`` (area density) and
    ``sum_i X_i`` against ``Q`` (cell total).
    Units: ``A_i`` [m2]; ``phi`` keeps the input units; ``Q`` keeps the cell-total units.
    """

    def __init__(
        self,
        name: str | None = None,
        aggregation_strategy: str | None = None,
        preserve_totals: bool | None = None,
        weighting_type: str | None = None,
        device=None,
        *,
        source_scale: str,
        target_scale: str,
        config=None,
        parameter_manager=None,
        **kwargs,
    ):
        metadata = FunctorMetadata(
            name=name or f"SpatialAggregation_{source_scale}_to_{target_scale}",
            version="1.0.0",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            description="Whole-support spatial aggregation on the declared basis with a ledger audit.",
            references=[
                "Jones, P.W. (1999). Conservative remapping schemes",
                "CF-1.8 area-weighted spatial aggregation guidance",
            ],
            mathematical_expression=(
                "intensive, area_density: phi = sum_i x_i*w_i / sum_i w_i (sum_i w_i > 0); "
                "cell_total: Q = sum_i X_i; categorical: argmax_c sum_{i: x_i = c} w_i; "
                "ledger sum_i x_i*A_i = phi*sum_i A_i (area density), sum_i X_i = Q (cell total)"
            ),
            parameters={
                "aggregation_strategy": "Whole-support law selector [dimensionless]",
                "preserve_totals": "Raise when an amount ledger residual exceeds the tolerance [dimensionless]",
                "weighting_type": "Spatial weighting rule selector [dimensionless]",
                "conservation_tolerance": "Relative ledger tolerance from coupling.scales.defaults [dimensionless]",
            },
            input_variables=["fine_grid_fields"],
            output_variables=["coarse_grid_fields"],
            optional_output_variables=["coarse_grid_fields"],
            conservation_laws=[],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        ScaleCouplingFunctor.__init__(
            self,
            name=name or metadata.name,
            metadata=metadata,
            source_scale=source_scale,
            target_scale=target_scale,
            coupling_type="spatial",
            conservation_properties=[],
            device=device
            or (device_policy.resolve_device(default="auto")),
            config=config or {},
            parameter_manager=parameter_manager,
        )
        # The logical input port names the call-time field set (kwargs["variables"] or every
        # tensor in state); it is not a literal state key, so validate_state must not require it.
        self.optional_input_variables = ("fine_grid_fields",)
        param_ns = SPATIAL_NS
        strategy_value = aggregation_strategy
        if strategy_value is None:
            strategy_value = self.get_parameter_value(
                "aggregation_strategy",
                namespace=param_ns,
                expected_type=str,
                units="dimensionless",
                required=True,
                validate=True,
            )
        strategy_value = str(strategy_value).lower()
        if strategy_value not in _AGGREGATION_SELECTORS:
            raise ValueError(
                f"Unsupported aggregation_strategy {strategy_value!r}; accepted selectors are "
                f"{sorted(_AGGREGATION_SELECTORS)} (each names a whole-support basis law; no "
                "maximum, minimum or median law exists)."
            )

        preserve_totals_value = bool(preserve_totals) if preserve_totals is not None else bool(
            self.get_parameter_value(
                "preserve_totals",
                namespace=param_ns,
                expected_type=bool,
                units="dimensionless",
                required=True,
                validate=True,
            )
        )

        weighting_value = weighting_type
        if weighting_value is None:
            weighting_value = self.get_parameter_value(
                "weighting_type",
                namespace=param_ns,
                expected_type=str,
                units="dimensionless",
                required=True,
                validate=True,
            )
        if weighting_value != "area":
            raise ValueError(
                f"Unsupported weighting_type {weighting_value!r}: the whole-support law weights "
                "with area measures only (coupling.scales.spatial.weighting_type allows 'area')."
            )

        tolerance = self.get_parameter_value(
            "conservation_tolerance",
            namespace=DEFAULT_NS,
            units="dimensionless",
            validate=True,
            required=True,
        )
        if isinstance(tolerance, Tensor):
            tolerance = float(tolerance.detach().cpu().item())
        self.conservation_tolerance = float(tolerance)
        if self.conservation_tolerance <= 0.0:
            raise ValueError("conservation_tolerance must be positive.")

        self.aggregation_strategy = strategy_value
        self.reduction_strategy: Optional[ScalingStrategy] = _AGGREGATION_SELECTORS[strategy_value]
        self.preserve_totals = preserve_totals_value
        self.weighting_type = weighting_value
        self.device = device or (
            device_policy.resolve_device(default="auto")
        )
        self.last_scaling_audit: Optional[ScalingAudit] = None
        self.parameter_manager = self.get_parameter_manager()
        self._facade = kwargs.get("_facade", None)
        self.metadata.conservation_laws = ["mass", "energy"] if self.preserve_totals else []
        self.conservation_properties = list(self.metadata.conservation_laws)
        self.parameter_values = {
            "aggregation_strategy": self.aggregation_strategy,
            "preserve_totals": self.preserve_totals,
            "weighting_type": self.weighting_type,
            "conservation_tolerance": self.conservation_tolerance,
        }

    def forward(self, state: dict, time: float, dt: float, **kwargs):
        """
        Reduce each requested field to one value according to its declared spatial basis.
        """
        # References: Jones (1999) and CF-1.8 cell_methods ("area: mean", "area: sum",
        #   "area: mode") as cited in the class docstring; theory scale_coupling.md sections 2.2
        #   and 2.3.2; the shared law and ledger aggregate_spatial_alias
        #   (relic/coupling/scales/alias_utils.py); RELIC ScaleCoupling contract
        #   (relic/core/functor/scale_coupling_bridge.py).
        # Mathematical expression: over the flattened cell index set i of each field
        #   intensive, area_density  phi = sum_i x_i w_i / sum_i w_i   (w_i = weights; 1 for an
        #                                                              intensive field without)
        #   cell_total               Q   = sum_i X_i                     (weights not applied)
        #   categorical              c*  = argmax_c sum_{i: x_i = c} w_i
        #   ledger of an amount      sum_i x_i A_i = phi sum_i A_i, sum_i X_i = Q
        # Parameter meanings: x_i/X_i source cell value; w_i caller weight (the cell area A_i of
        #   an area density); phi, Q, c* the value of the whole support, emitted as
        #   "<var>_aggregated" in the input basis.
        # Units: w_i and A_i [m2] for area weighting; phi keeps the input units (an area density
        #   stays an amount per unit area); Q keeps the cell-total units; the ledger is an amount.
        # Domain/constraints: an empty variable list returns {}; every requested variable must be
        #   a non-empty tensor of any rank; the basis must be declared (no name heuristic);
        #   weights must have exactly each field's shape, be finite and non-negative; area
        #   densities need weights; a mean or a mode needs sum_i w_i > 0; the selector must name
        #   the basis law of every variable; with preserve_totals a ledger residual above the
        #   accepted residual raises (otherwise it is reported); nothing is clamped or rescaled.
        # Sign convention: the sign of the source field is preserved; weights are non-negative.
        variables = kwargs.get("variables") or [k for k, v in state.items() if isinstance(v, Tensor)]
        if not variables:
            return {}
        area_weights = kwargs.get("weights")
        if area_weights is not None and not isinstance(area_weights, Tensor):
            raise TypeError("SpatialAggregation weights must be a tensor with the field's shape.")
        variable_metadata = kwargs.get("variable_metadata")
        fields: Dict[str, Tensor] = {}
        bases: Dict[str, SpatialQuantityBasis] = {}
        for var in variables:
            tensor = state.get(var)
            if not isinstance(tensor, Tensor):
                raise ValueError(f"SpatialAggregation variable {var!r} is not a tensor in state.")
            basis, source = _resolve_variable_basis(var, self, variable_metadata)
            logger.debug("SpatialAggregation basis for %s: %s (from %s)", var, basis.value, source)
            fields[var] = tensor
            bases[var] = basis
        result, audit = aggregate_spatial_alias(
            data_dict=fields,
            variable_bases=bases,
            source_scale=self.source_scale,
            target_scale=self.target_scale,
            mode=ScalingMode.CONSERVATIVE,
            device=self.device,
            enforce_conservation=self.preserve_totals,
            tolerance=self.conservation_tolerance,
            weights=area_weights,
            strategy=self.reduction_strategy,
        )
        self.last_scaling_audit = audit
        return {f"{var}_aggregated": value for var, value in result.items()}

    def regrid(self, data: Tensor, source_grid=None, target_grid=None, **kwargs) -> Tensor:
        """SpatialAggregation reduces fields; it defines no grid-to-grid regridding."""
        raise NotImplementedError(
            "SpatialAggregation has no grid-to-grid regrid; use ConservativeRemap or the "
            "spatial regridder factory for scale changes between grids."
        )

    def upscale(self, fine_data: Tensor, **kwargs) -> Tensor:
        """Reduce one field (``variables=[name]``) through forward with its declared basis."""
        variables = kwargs.get("variables") or ["data"]
        state = kwargs.get("state") or {variables[0]: fine_data}
        result = self.forward(
            state=state,
            time=kwargs.get("time", 0.0),
            dt=kwargs.get("dt", 1.0),
            variables=variables,
            weights=kwargs.get("weights"),
            variable_metadata=kwargs.get("variable_metadata"),
        )
        return result[f"{variables[0]}_aggregated"]

    def downscale(self, coarse_data: Tensor, **kwargs) -> Tensor:
        """SpatialAggregation defines no disaggregation; it raises instead of returning the input."""
        raise NotImplementedError(
            "SpatialAggregation does not implement downscale (disaggregation); use the "
            "hierarchical disaggregation or ConservativeRemap.downscale for scale changes."
        )


class ConservativeRemap(SpatialScalingBase, ScaleCouplingFunctor):
    """
    First-order conservative remapping on explicit overlap geometry.

    References:
        - Jones, P.W. (1999). First- and second-order conservative remapping schemes
          for grids in spherical coordinates. Mon. Weather Rev. 127:2204-2210.
        - Jones, P.W. (1998). SCRIP user guide: "fracarea" and "destarea"
          normalization of conservative weights.
    Mathematical expression:
        A_iJ is the overlap area of source cell i with target cell J, A_i the source
        area, A_J the target area and C_J = sum_i A_iJ the covered target area. Every
        variable carries one explicit SpatialQuantityBasis (relic.coupling.scales.types):
        cell_total:   Y_J = sum_i X_i A_iJ / A_i
        area_density: y_J = sum_i x_i A_iJ / A_J (via X_i = x_i A_i and y_J = Y_J / A_J)
        intensive:    y_J = sum_i x_i A_iJ / C_J
        downscale (transposed overlap) swaps the roles of A_i and A_J.
        The source amount outside the target support, U = sum_i X_i (1 - f_i) with
        f_i = sum_J A_iJ / A_i, is reported as ``<var>_uncovered_source_amount``.
    Units:
        A_iJ, A_i, A_J, C_J [m2]; X, Y, U [amount]; x, y of an area density [amount m-2];
        intensive values keep their input units.
    Domain/constraints:
        A_iJ >= 0, A_i > 0, sum_J A_iJ <= A_i (1 + tol); intensive targets need C_J > 0;
        area densities need A_J; the basis comes from ``variable_metadata`` or the YAML
        variable catalogs and a bare ``extensive`` or undeclared variable raises;
        geometry must be supplied as ``overlap_areas`` with ``source_areas`` (and
        ``target_areas`` for downscaling or area densities) or as
        ``source_grid``/``target_grid`` resolved through the canonical
        ConservativeRegridder; there is no aggregate fallback. ``source_grid_fields`` and
        ``target_grid_fields`` are logical port names for the call-time field set named
        by ``variables``; FunctorBase treats them as optional state keys so the governed
        call ``functor(state, time, dt, ...)`` runs the same forward.
    Sign convention:
        values keep their sign; with ``preserve_positivity`` negative outputs are clipped
        to zero and the clip residual (StateManager clip_loss sign: requested minus clipped
        amount, negative for the material the clip adds) is disclosed as
        ``diagnostics.conservative_remap.positivity_clip_residual`` and in the audit.
    """

    DIAGNOSTIC_KEYS = (
        "diagnostics.conservative_remap.mass_balance_error",
        "diagnostics.conservative_remap.relative_error",
        "diagnostics.conservative_remap.covered_source_fraction",
        "diagnostics.conservative_remap.positivity_clip_residual",
    )

    def __init__(
        self,
        preserve_mass: bool = True,
        preserve_energy: bool = True,
        preserve_positivity: bool = False,
        device=None,
        **kwargs,
    ):
        metadata = FunctorMetadata(
            name="conservative_remap",
            version="2.0.0",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            description="First-order conservative remap on explicit overlap areas (Jones 1999).",
            references=[
                "Jones, P.W. (1999). Conservative remapping schemes for climate models",
                "Jones, P.W. (1998). SCRIP user guide, fracarea and destarea normalization",
            ],
            mathematical_expression=(
                "intensive y_J = sum_i x_i A_iJ / sum_i A_iJ; cell_total Y_J = sum_i X_i A_iJ / A_i; "
                "area_density y_J = sum_i x_i A_iJ / A_J; A_iJ overlap area [m2]"
            ),
            parameters={
                "preserve_mass": "Enable strict mass conservation checks [dimensionless]",
                "preserve_energy": "Enable strict energy conservation checks [dimensionless]",
                "preserve_positivity": "Clamp remapped outputs to non-negative values and disclose the clipped residual [dimensionless]",
                "conservation_tolerance": "Relative conservation tolerance from coupling.scales.defaults [dimensionless]",
            },
            input_variables=["source_grid_fields"],
            output_variables=["target_grid_fields", *ConservativeRemap.DIAGNOSTIC_KEYS],
            optional_output_variables=["target_grid_fields"],
            conservation_laws=[
                law for law, flag in (("mass", preserve_mass), ("energy", preserve_energy)) if flag
            ],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        ScaleCouplingFunctor.__init__(
            self,
            name=metadata.name,
            metadata=metadata,
            source_scale="source_grid",
            target_scale="target_grid",
            coupling_type="spatial",
            conservation_properties=[
                prop for prop, flag in (("mass", preserve_mass), ("energy", preserve_energy)) if flag
            ],
            device=device
            or (device_policy.resolve_device(default="auto")),
        )
        # The logical input port names the call-time field set (kwargs["variables"]); it is
        # not a literal state key, so FunctorBase.validate_state must not require it.
        self.optional_input_variables = ("source_grid_fields",)
        self.preserve_positivity = preserve_positivity
        self.preserve_mass = preserve_mass
        self.preserve_energy = preserve_energy
        self.conservation_properties = []
        if preserve_mass:
            self.conservation_properties.append("mass")
        if preserve_energy:
            self.conservation_properties.append("energy")
        tolerance = self.get_parameter_value(
            "conservation_tolerance",
            namespace=DEFAULT_NS,
            units="dimensionless",
            validate=True,
            required=True,
        )
        if isinstance(tolerance, Tensor):
            tolerance = float(tolerance.detach().cpu().item())
        self.conservation_tolerance = float(tolerance)
        if self.conservation_tolerance <= 0.0:
            raise ValueError("conservation_tolerance must be positive.")
        self.last_scaling_audit: Optional[ScalingAudit] = None
        # Overlap geometry keyed by the content fingerprint of both descriptors and the tolerance
        # (grid_utils.content_fingerprint), never by object identity.
        self._geometry_cache: Dict[str, Tuple[Tensor, Tensor, Tensor]] = {}

    # ------------------------------------------------------------------ geometry
    def _overlap_from_grids(self, source_grid, target_grid) -> Tuple[Tensor, Tensor, Tensor]:
        """Resolve A_iJ, A_i and A_J from grid descriptors through ConservativeRegridder."""
        # References: relic/coupling/scales/spatial/regrid_conservative.py (canonical
        #   overlap computation for regular and polygon grids).
        # Mathematical expression: the regridder stores overlap fractions W_iJ = A_iJ / A_i on
        #   its dense and sparse paths alike (rows are not renormalized), so A_iJ = W_iJ A_i.
        # Parameter meanings: W_iJ regridder weight; A_i, A_J cell areas from the descriptors.
        # Units: A [m2]; W dimensionless.
        # Domain/constraints: both descriptors are required; only the regridder's geometry is
        #   used here, so no quantity basis is declared on it. The geometry is reused only for
        #   descriptors of equal content (every GridDescriptor field) and the same tolerance: a
        #   descriptor changed in place, or a new one that reuses a freed object's identity, is
        #   computed again.
        # Sign convention: areas are non-negative.
        from relic.coupling.scales.spatial.grid_utils import content_fingerprint

        key = content_fingerprint(source_grid, target_grid, self.conservation_tolerance)
        cached = self._geometry_cache.get(key)
        if cached is not None:
            return cached
        from relic.coupling.scales.spatial.regrid_conservative import ConservativeRegridder

        regridder = ConservativeRegridder(
            source_grid,
            target_grid,
            conservation_tolerance=self.conservation_tolerance,
            use_sparse_weights=False,
            device=self.device,
            parameter_manager=self.get_parameter_manager(),
        )
        weights = regridder.weight_matrix
        source_areas = regridder.source_areas.reshape(-1).to(device=self.device, dtype=torch.float64)
        target_areas = regridder.target_areas.reshape(-1).to(device=self.device, dtype=torch.float64)
        dense_weights = weights.to_dense() if weights.is_sparse else weights
        overlap = dense_weights.to(device=self.device, dtype=torch.float64) * source_areas.unsqueeze(1)
        geometry = (overlap, source_areas, target_areas)
        self._geometry_cache[key] = geometry
        return geometry

    def _resolve_geometry(self, kwargs: Dict[str, Any]) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        overlap = kwargs.get("overlap_areas")
        source_areas = kwargs.get("source_areas")
        target_areas = kwargs.get("target_areas")
        source_grid = kwargs.get("source_grid")
        target_grid = kwargs.get("target_grid")
        if overlap is None and (source_grid is not None or target_grid is not None):
            if source_grid is None or target_grid is None:
                raise ValueError("ConservativeRemap requires both source_grid and target_grid.")
            overlap, source_areas, target_areas = self._overlap_from_grids(source_grid, target_grid)
        if overlap is None:
            raise ValueError(
                "ConservativeRemap requires overlap_areas [n_source, n_target] with source_areas "
                "(or source_grid and target_grid); no aggregate fallback exists."
            )
        if source_areas is None:
            raise ValueError("source_areas [n_source] are required together with overlap_areas.")
        return overlap, source_areas, target_areas

    def _variable_bases(
        self,
        variables: List[str],
        variable_metadata: Optional[Dict[str, Dict[str, Any]]],
    ) -> Dict[str, SpatialQuantityBasis]:
        """Resolve one explicit SpatialQuantityBasis per variable (call metadata or YAML catalog)."""
        bases: Dict[str, SpatialQuantityBasis] = {}
        for var in variables:
            basis, source = _resolve_variable_basis(var, self, variable_metadata)
            logger.debug("ConservativeRemap basis for %s: %s (from %s)", var, basis.value, source)
            bases[var] = basis
        return bases

    def _finalize(
        self,
        result: Dict[str, Tensor],
        audit: ScalingAudit,
        suffix: str,
        *,
        variable_bases: Dict[str, SpatialQuantityBasis],
        overlap: Tensor,
        target_areas: Optional[Tensor],
    ) -> Dict[str, Tensor]:
        """Apply positivity clipping with residual accounting and expose the audit."""
        # References: docs/source/theory/scale_coupling.md section 2.3 (positivity bounds);
        #   relic/core/state/manager.py bound enforcement (clip_loss = requested - clipped).
        # Mathematical expression: y_c = max(y, 0); C = sum_J (y_J - y_c,J) a_J with a_J = 1 for
        #   cell totals, A_J for area densities and C_J = sum_i A_iJ for intensive fields, so the
        #   ledger amount before the clip = the amount after the clip + C.
        # Parameter meanings: y remapped values; y_c clipped values; a_J the ledger measure of
        #   target cell J (target area A_J, covered area C_J); C clip residual of one variable.
        # Units: C in the ledger amount of the field (the cell-total unit, value x m2 for area
        #   densities and intensive fields); A_J and C_J [m2].
        # Domain/constraints: C <= 0 (only negative values are raised to 0); C < 0 breaks strict
        #   conservation and is recorded as an audit warning instead of being hidden; the reported
        #   diagnostic is the sum of C over the call's variables.
        # Sign convention: StateManager clip_loss sign: C < 0 is the amount the clip added.
        outputs: Dict[str, Tensor] = {}
        clip_residual = 0.0
        covered = torch.as_tensor(overlap).to(device=self.device, dtype=torch.float64).sum(dim=0)
        for key, value in result.items():
            remapped = value
            if self.preserve_positivity:
                clipped = torch.clamp(remapped, min=0.0)
                basis = SpatialQuantityBasis.parse(variable_bases[key])
                if basis is SpatialQuantityBasis.CELL_TOTAL:
                    measure = torch.ones_like(covered)
                elif basis is SpatialQuantityBasis.AREA_DENSITY:
                    measure = torch.as_tensor(target_areas).to(device=self.device, dtype=torch.float64).reshape(-1)
                else:
                    measure = covered
                delta = (remapped.to(torch.float64) - clipped.to(torch.float64)).reshape(-1)
                residual = float((delta * measure).sum().item())
                if residual < 0.0:
                    clip_residual += residual
                    audit.warnings.append(
                        f"positivity clipping added {-residual:.6e} to {key} (clip residual {residual:.6e}); "
                        "conservation is broken by that amount"
                    )
                remapped = clipped
            outputs[f"{key}{suffix}"] = remapped
            # The source amount outside the target support is emitted per variable so that a
            # caller needing whole-domain closure can route it explicitly (theory 2.3).
            outputs[f"{key}_uncovered_source_amount"] = torch.tensor(
                float(audit.metadata["uncovered_source_amount"][key]),
                dtype=torch.float64,
                device=self.device,
            )
        audit.metadata["positivity_clip_residual"] = clip_residual
        audit.physical_constraints_applied = bool(self.preserve_positivity)
        self.last_scaling_audit = audit
        diagnostics = {
            "diagnostics.conservative_remap.mass_balance_error": audit.mass_balance_error,
            "diagnostics.conservative_remap.relative_error": audit.relative_error,
            "diagnostics.conservative_remap.covered_source_fraction": audit.metadata["covered_source_fraction"],
            "diagnostics.conservative_remap.positivity_clip_residual": clip_residual,
        }
        for key, value in diagnostics.items():
            outputs[key] = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        return outputs

    def forward(self, state: dict, time: float, dt: float, **kwargs):
        """
        Perform overlap-weighted conservative remapping for the requested variables.
        """
        # References: Jones (1999); SCRIP fracarea/destarea normalization; RELIC ScaleCoupling
        #   contract (relic/core/functor/scale_coupling_bridge.py); theory scale_coupling.md 2.3.
        # Mathematical expression: remap_conservative_alias evaluates, per explicit basis,
        #   cell_total Y_J = sum_i X_i A_iJ / A_i, area_density y_J = sum_i x_i A_iJ / A_J and
        #   intensive y_J = sum_i x_i A_iJ / sum_i A_iJ on the resolved geometry, and reports
        #   U = sum_i X_i (1 - f_i); with preserve_positivity y <- max(y, 0) and the clipped
        #   residual is disclosed.
        # Parameter meanings: x_i/X_i source value; A_iJ overlap area; A_i source area; A_J
        #   target area; f_i covered fraction of source i; basis from variable_metadata or the
        #   YAML variable catalogs (no name heuristic); outputs "<var>_remapped",
        #   "<var>_uncovered_source_amount" and the four diagnostics.conservative_remap.* scalars.
        # Units: fields keep their input units and basis; A [m2]; U in the amount unit;
        #   diagnostics: mass_balance_error in the ledger units, relative_error and
        #   covered_source_fraction [1].
        # Domain/constraints: an empty variable list returns {}; a requested variable that is
        #   not a tensor, an undeclared variable or a bare "extensive" declaration raises;
        #   geometry is mandatory; when preserve_mass or preserve_energy is set a relative
        #   covered-support conservation error above conservation_tolerance raises RuntimeError.
        # Sign convention: values keep their sign unless preserve_positivity clips them.
        variables = kwargs.get("variables") or [k for k, v in state.items() if isinstance(v, Tensor)]
        if not variables:
            return {}
        missing = [k for k in variables if not isinstance(state.get(k), Tensor)]
        if missing:
            raise ValueError(f"ConservativeRemap variables are not tensors in state: {missing}.")
        overlap, source_areas, target_areas = self._resolve_geometry(kwargs)
        variable_bases = self._variable_bases(variables, kwargs.get("variable_metadata"))
        result, audit = remap_conservative_alias(
            data_dict={k: state[k] for k in variables},
            overlap_areas=overlap,
            source_areas=source_areas,
            variable_bases=variable_bases,
            source_scale=self.source_scale,
            target_scale=self.target_scale,
            mode=ScalingMode.CONSERVATIVE,
            device=self.device,
            enforce_conservation=bool(self.conservation_properties),
            tolerance=self.conservation_tolerance,
            operation="remap",
            target_areas=target_areas,
        )
        return self._finalize(
            result, audit, "_remapped",
            variable_bases=variable_bases, overlap=overlap, target_areas=target_areas,
        )

    def get_dependencies(self) -> List[str]:
        """No static upstream dependency; instance declarations (add_dependency,
        set_dependencies, config["dependencies"]) are merged by the FunctorBase owner."""
        return self._merge_declared_dependencies([])

    def regrid(self, data: Tensor, source_grid=None, target_grid=None, *, method: str, preserve: str = "mass", **kwargs) -> Tensor:
        """Conservative regrid of one field between grid descriptors.

        ``quantity_basis`` (a SpatialQuantityBasis or its string value) is required: the
        ``preserve`` label does not say whether the field is a cell total or an area density.
        ``preserve`` must be ``mass``, ``energy`` or ``none``; the conservation check itself
        follows the preserve_mass/preserve_energy policy of this instance.
        """
        if method != "conservative":
            raise ValueError(f"ConservativeRemap only implements the conservative method, got {method!r}.")
        if preserve not in ("mass", "energy", "none"):
            raise ValueError(f"preserve must be 'mass', 'energy' or 'none', got {preserve!r}.")
        if "quantity_basis" not in kwargs:
            raise ValueError(
                "ConservativeRemap.regrid requires quantity_basis (intensive, cell_total or area_density)."
            )
        basis = SpatialQuantityBasis.parse(kwargs["quantity_basis"])
        result = self.forward(
            {"data": data},
            kwargs.get("time", 0.0),
            kwargs.get("dt", 1.0),
            variables=["data"],
            source_grid=source_grid,
            target_grid=target_grid,
            variable_metadata={"data": {"spatial_basis": basis.value}},
        )
        return result["data_remapped"]

    def upscale(self, fine_data: Tensor, **kwargs) -> Tensor:
        """Source-to-target remap of one field; geometry is passed through kwargs."""
        variables = kwargs.pop("variables", None) or ["data"]
        state = kwargs.pop("state", None) or {variables[0]: fine_data}
        time = kwargs.pop("time", 0.0)
        dt = kwargs.pop("dt", 1.0)
        result = self.forward(state=state, time=time, dt=dt, variables=variables, **kwargs)
        return result[f"{variables[0]}_remapped"]

    def downscale(self, coarse_data: Tensor, **kwargs) -> Tensor:
        """Target-to-source remap of one field on the transposed overlap geometry."""
        # References: Jones (1999); theory scale_coupling.md 2.3 (transposed overlap).
        # Mathematical expression: with the transposed overlap A^T (target rows, source
        #   columns) and the target areas A_J as the "source" areas of the transposed problem:
        #   intensive x_i = sum_J y_J A_iJ / sum_J A_iJ; cell_total X_i = sum_J Y_J A_iJ / A_J;
        #   area_density x_i = sum_J y_J A_iJ / A_i.
        # Parameter meanings: y_J/Y_J target values; A_iJ overlap area; A_J target area; A_i
        #   source area (the target area of the transposed problem).
        # Units: A [m2]; x keeps the units and basis of y.
        # Domain/constraints: target_areas are mandatory (or grids); the coverage checks
        #   sum_i A_iJ <= A_J (1 + tol) and sum_J A_iJ <= A_i (1 + tol) and the conservation
        #   check apply as in forward; the basis must be declared.
        # Sign convention: values keep their sign unless preserve_positivity clips them.
        variables = kwargs.pop("variables", None) or ["data"]
        if len(variables) != 1:
            raise ValueError("downscale remaps exactly one field; pass variables=[name].")
        overlap, source_areas, target_areas = self._resolve_geometry(kwargs)
        if target_areas is None:
            raise ValueError("downscale requires target_areas [n_target] (or source_grid and target_grid).")
        variable_bases = self._variable_bases(variables, kwargs.get("variable_metadata"))
        overlap_t = torch.as_tensor(overlap).to(device=self.device, dtype=torch.float64).transpose(0, 1)
        result, audit = remap_conservative_alias(
            data_dict={variables[0]: coarse_data},
            overlap_areas=overlap_t,
            source_areas=target_areas,
            variable_bases=variable_bases,
            source_scale=self.target_scale,
            target_scale=self.source_scale,
            mode=ScalingMode.CONSERVATIVE,
            device=self.device,
            enforce_conservation=bool(self.conservation_properties),
            tolerance=self.conservation_tolerance,
            operation="downscale",
            target_areas=source_areas,
        )
        outputs = self._finalize(
            result, audit, "_downscaled",
            variable_bases=variable_bases, overlap=overlap_t, target_areas=source_areas,
        )
        return outputs[f"{variables[0]}_downscaled"]


# Stable aliases for the remaining spatial implementations
SpatialScaleCoupling = SpatialAggregation
GridRemapping = ConservativeRemap
ConservativeGridRemapping = ConservativeRemap


def aggregate_spatial(
    data: Dict[str, Tensor],
    strategy: str = "area_weighted",
    *,
    source_scale: str,
    target_scale: str,
    **kwargs,
) -> Dict[str, Tensor]:
    """
    Functional API: run spatial aggregation without instantiating the functor. The source and
    target scale of the aggregation are required, as for SpatialAggregation.
    """
    aggregator = SpatialAggregation(
        source_scale=source_scale,
        target_scale=target_scale,
        aggregation_strategy=strategy,
        preserve_totals=kwargs.get("preserve_totals"),
        weighting_type=kwargs.get("weighting_type"),
        device=kwargs.get("device"),
    )
    return aggregator.forward(
        state=data,
        time=kwargs.get("time", 0.0),
        dt=kwargs.get("dt", 1.0),
        variables=kwargs.get("variables"),
        weights=kwargs.get("weights"),
        variable_metadata=kwargs.get("variable_metadata"),
    )


def conservative_remap(
    data: Dict[str, Tensor],
    source_grid=None,
    target_grid=None,
    preserve_mass: bool | None = None,
    preserve_energy: bool | None = None,
    preserve_positivity: bool | None = None,
    device=None,
    variables: Optional[List[str]] = None,
    **kwargs,
) -> Dict[str, Tensor]:
    """
    Functional API: run overlap-weighted conservative remapping without instantiating the functor.

    Geometry is passed through ``kwargs`` (``overlap_areas``/``source_areas``/``target_areas``
    or ``source_grid``/``target_grid``) together with ``variable_metadata``.
    """
    remapper = ConservativeRemap(
        preserve_mass=preserve_mass if preserve_mass is not None else True,
        preserve_energy=preserve_energy if preserve_energy is not None else True,
        preserve_positivity=preserve_positivity if preserve_positivity is not None else False,
        device=device,
    )
    time = kwargs.pop("time", 0.0)
    dt = kwargs.pop("dt", 1.0)
    return remapper.forward(
        state=data,
        time=time,
        dt=dt,
        variables=variables,
        source_grid=source_grid,
        target_grid=target_grid,
        **kwargs,
    )
