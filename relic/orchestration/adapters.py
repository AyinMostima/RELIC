"""
Adapter registry for orchestration.

Adapters are used to bridge unit and semantic mismatches between ports.
They must not hardcode parameter values; all context is provided externally.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

import torch
from torch import Tensor

from ..core.functor import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    ScaleCouplingFunctor,
    SpatialScale,
    VariableKind,
)
from ..core.state import StateWriteMode, UpdateItem, UpdateMode
from ..coupling.core.policy import ScalingMode
from ..coupling.scales.alias_utils import aggregate_spatial_alias
from ..coupling.scales.types import ScalingStrategy, SpatialQuantityBasis
from ..data.standards.unit_converter import (
    ConversionContext,
    UnitConverter,
)
from ..data.standards.unit_converter import (
    unit_converter as canonical_unit_converter,
)
from ..parameters.namespaces import ParameterNamespace
from ..utils.physical_constants import get_physical_constant
from .port_spec import NodeDefinition, PortSpec

if TYPE_CHECKING:
    from .node_registry import NodeRegistry


@dataclass
class AdapterResult:
    """Adapter application result."""

    value: Any
    metadata: Dict[str, Any]


class AdapterBase:
    """Base adapter interface."""

    adapter_type: str = "adapter"

    def apply(self, value: Any, **kwargs: Any) -> AdapterResult:
        raise NotImplementedError("AdapterBase.apply must be implemented.")


class UnitCastAdapter(AdapterBase):
    """Unit conversion adapter using UnitConverter."""

    adapter_type = "unit_cast"

    def __init__(self, unit_converter: Optional[UnitConverter] = None) -> None:
        self.unit_converter = (
            unit_converter
            if unit_converter is not None
            else canonical_unit_converter
        )

    def apply(
        self,
        value: Any,
        variable_name: str,
        source_units: str,
        target_units: str,
        context: Optional[ConversionContext] = None,
    ) -> AdapterResult:
        """
        References:
            CF-1.8 unit conventions; RELIC UnitConverter mappings.
        Mathematical Expression:
            x_out = convert(x_in, source_units, target_units)
        Parameters:
            value: input value; variable_name: standard name; source_units: str; target_units: str; context: ConversionContext.
        """
        converted = self.unit_converter.convert(
            value,
            variable_name=variable_name,
            source_units=source_units,
            target_units=target_units,
            context=context,
        )
        return AdapterResult(
            value=converted,
            metadata={
                "adapter_type": self.adapter_type,
                "variable_name": variable_name,
                "source_units": source_units,
                "target_units": target_units,
            },
        )


class TemporalResampleAdapter(AdapterBase):
    """Temporal resampling adapter with explicit conservative reductions."""

    adapter_type = "temporal_resample"

    def apply(self, value: Any, **kwargs: Any) -> AdapterResult:
        """
        References:
            CF-1.8 cell_methods; conservative aggregation references;
            Astrom and Murray (2008), sampled-data systems.
        Mathematical Expression:
            Window reduction: y = sum_i(dt_i*x_i) / sum_i(dt_i);
            current-step ZOH: x(tau) = x(t_n) on [t_n, t_n+dt), so
            mean(x) = x(t_n).
        Parameters:
            value: source samples or current-step value; sample_durations:
            sample widths or the current observation-window duration; dt:
            actual simulator step; sampling_mode: window or current_step.
        """
        sampling_mode = str(
            kwargs.get("sampling_mode", "window")
        ).strip().lower()
        if sampling_mode not in {"window", "current_step"}:
            raise ValueError(
                "TemporalResampleAdapter sampling_mode must be 'window' "
                "or 'current_step'."
            )
        resampler = kwargs.get("temporal_resampler")
        if resampler is not None:
            if sampling_mode != "window":
                raise ValueError(
                    "A callable temporal_resampler is incompatible with "
                    "sampling_mode='current_step'."
                )
            result = resampler(value)
            return AdapterResult(
                value=result,
                metadata={
                    "adapter_type": self.adapter_type,
                    "method": "callable",
                },
            )

        method = str(kwargs.get("method", "")).strip().lower()
        if method not in {"interval_mean", "interval_integral"}:
            raise RuntimeError(
                "TemporalResampleAdapter requires temporal_resampler or "
                "method='interval_mean|interval_integral'."
            )

        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if not torch.is_floating_point(tensor):
            tensor = tensor.to(torch.float32)
        if sampling_mode == "current_step":
            if method != "interval_mean":
                raise ValueError(
                    "sampling_mode='current_step' only supports "
                    "method='interval_mean'."
                )
            if "axis" in kwargs:
                raise ValueError(
                    "sampling_mode='current_step' must not declare axis."
                )
            sample_duration = kwargs.get("sample_durations")
            if sample_duration is None:
                raise RuntimeError(
                    "sampling_mode='current_step' requires an explicit "
                    "observation-window duration."
                )
            duration = torch.as_tensor(
                sample_duration,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            runtime_dt = kwargs.get("dt")
            if runtime_dt is None:
                raise RuntimeError(
                    "sampling_mode='current_step' requires the runtime dt."
                )
            step_duration = torch.as_tensor(
                runtime_dt,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            if duration.ndim != 0:
                raise ValueError(
                    "Current-step observation-window duration must be scalar."
                )
            if step_duration.ndim != 0:
                raise ValueError("Runtime dt must be scalar.")
            if not bool(torch.isfinite(duration)) or not bool(duration > 0):
                raise ValueError(
                    "Current-step observation-window duration must be finite "
                    "and positive."
                )
            if not bool(torch.isfinite(step_duration)) or not bool(
                step_duration > 0
            ):
                raise ValueError("Runtime dt must be finite and positive.")
            if not bool(torch.equal(duration, step_duration)):
                raise ValueError(
                    "Current-step observation-window duration must equal the "
                    "runtime dt."
                )
            return AdapterResult(
                value=tensor,
                metadata={
                    "adapter_type": self.adapter_type,
                    "sampling_mode": sampling_mode,
                    "method": method,
                    "duration_sum": float(duration.detach().cpu().item()),
                    "reconstruction": "zero_order_hold",
                },
            )

        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        axis = int(kwargs.get("axis", 0))
        if axis < 0:
            axis += tensor.ndim
        if axis < 0 or axis >= tensor.ndim:
            raise ValueError(
                f"TemporalResampleAdapter axis {axis} is invalid for "
                f"{tensor.ndim}-D input."
            )

        sample_durations = kwargs.get("sample_durations")
        if sample_durations is None:
            durations = torch.ones(
                tensor.shape[axis],
                dtype=tensor.dtype,
                device=tensor.device,
            )
        else:
            durations = torch.as_tensor(
                sample_durations,
                dtype=tensor.dtype,
                device=tensor.device,
            )
        if durations.ndim != 1 or durations.shape[0] != tensor.shape[axis]:
            raise ValueError(
                "sample_durations must be one-dimensional and match the "
                "resampled axis length."
            )
        if not bool(torch.all(torch.isfinite(durations))):
            raise ValueError("sample_durations must be finite.")
        if bool(torch.any(durations < 0)):
            raise ValueError("sample_durations must be non-negative.")
        duration_sum = durations.sum()
        if not bool(duration_sum > 0):
            raise ValueError("sample_durations must have a positive sum.")

        weight_shape = [1] * tensor.ndim
        weight_shape[axis] = durations.shape[0]
        weighted = tensor * durations.reshape(weight_shape)
        result = weighted.sum(dim=axis)
        if method == "interval_mean":
            result = result / duration_sum
        return AdapterResult(
            value=result,
            metadata={
                "adapter_type": self.adapter_type,
                "method": method,
                "axis": axis,
                "duration_sum": float(duration_sum.detach().cpu().item()),
            },
        )


class SemanticAdapter(AdapterBase):
    """Semantic adapter for flux to increment conversions."""

    adapter_type = "semantic_adapter"

    def apply(self, value: Any, **kwargs: Any) -> AdapterResult:
        """
        References:
            Conservation law integration; mass balance relations.
        Mathematical Expression:
            delta = flux * dt
        Parameters:
            value: flux input; dt: time step in seconds.
        """
        dt = kwargs.get("dt")
        if dt is None:
            raise RuntimeError("SemanticAdapter requires dt in context.")
        result = value * dt
        return AdapterResult(
            value=result,
            metadata={
                "adapter_type": self.adapter_type,
                "dt": dt,
            },
        )


class AreaBasisTransformAdapter(AdapterBase):
    """Transform leaf-area and ground-area normalized flux densities."""

    adapter_type = "area_basis_transform"

    def apply(self, value: Any, **kwargs: Any) -> AdapterResult:
        """
        References:
            Guenther et al. (2012), MEGAN2.1 canopy emission formulation.
        Mathematical Expression:
            F_ground = F_leaf * LAI; F_leaf = F_ground / LAI.
        Parameters:
            value: flux density; leaf_area_index: leaf area per ground area;
            source_area_basis and target_area_basis: declared support bases.
        """
        source_basis = str(kwargs.get("source_area_basis", "")).strip().lower()
        target_basis = str(kwargs.get("target_area_basis", "")).strip().lower()
        allowed = {"leaf_area", "ground_area"}
        if source_basis not in allowed or target_basis not in allowed:
            raise ValueError(
                "source_area_basis and target_area_basis must be leaf_area "
                "or ground_area."
            )
        if source_basis == target_basis:
            raise ValueError("Area-basis adapter requires different bases.")

        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if not torch.is_floating_point(tensor):
            tensor = tensor.to(torch.float32)
        leaf_area_index = kwargs.get("leaf_area_index")
        if leaf_area_index is None:
            raise RuntimeError(
                "AreaBasisTransformAdapter requires leaf_area_index."
            )
        lai = torch.as_tensor(
            leaf_area_index,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        if not bool(torch.all(torch.isfinite(lai))):
            raise ValueError("leaf_area_index must be finite.")
        if bool(torch.any(lai < 0)):
            raise ValueError("leaf_area_index must be non-negative.")

        if source_basis == "leaf_area":
            result = tensor * lai
        else:
            if bool(torch.any(lai <= 0)):
                raise ValueError(
                    "leaf_area_index must be positive for ground-to-leaf "
                    "conversion."
                )
            result = tensor / lai
        return AdapterResult(
            value=result,
            metadata={
                "adapter_type": self.adapter_type,
                "source_area_basis": source_basis,
                "target_area_basis": target_basis,
            },
        )


class SurfaceFluxToVolumeSourceAdapter(AdapterBase):
    """Inject a molar surface flux into only the lowest volume layer."""

    adapter_type = "surface_flux_to_volume_source"

    @staticmethod
    def _as_float_tensor(value: Any, reference: Tensor) -> Tensor:
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        return tensor.to(dtype=reference.dtype, device=reference.device)

    @staticmethod
    def _normalize_axis(axis: int, ndim: int) -> int:
        normalized = axis + ndim if axis < 0 else axis
        if normalized < 0 or normalized >= ndim:
            raise ValueError(
                "Surface flux adapter vertical_axis is outside layer_volumes."
            )
        return normalized

    @staticmethod
    def _to_flat_list(value: Tensor) -> list[float]:
        return value.detach().cpu().reshape(-1).tolist()

    def apply(self, value: Any, **kwargs: Any) -> AdapterResult:
        """
        Convert an areal molar flux to a conservative mass-volume source.

        References:
            Patankar (1980), Numerical Heat Transfer and Fluid Flow;
            Crank (1975), The Mathematics of Diffusion; SI Brochure, 9th ed.
        Mathematical Expression:
            S[k, 0] = F[k] * M * A[k] / V[k, 0];
            S[k, j] = 0 for j > 0;
            sum_j(S[k, j] * V[k, j]) = F[k] * M * A[k].
        Parameters:
            F is surface molar flux [mol m-2 s-1]; M is molar mass
            [kg mol-1]; A is cell area [m2]; V is layer volume [m3];
            S is the derived volumetric mass source [kg m-3 s-1].
        """

        flux = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if not torch.is_floating_point(flux):
            flux = flux.to(torch.float32)
        raw_area = kwargs.get("cell_area")
        raw_volumes = kwargs.get("layer_volumes")
        if raw_area is None or raw_volumes is None:
            raise RuntimeError(
                "SurfaceFluxToVolumeSourceAdapter requires cell_area and "
                "layer_volumes."
            )
        cell_area = self._as_float_tensor(raw_area, flux)
        layer_volumes = self._as_float_tensor(raw_volumes, flux)
        if layer_volumes.ndim < 1:
            raise ValueError("layer_volumes must include a vertical dimension.")
        vertical_axis = self._normalize_axis(
            int(kwargs.get("vertical_axis", -1)),
            layer_volumes.ndim,
        )
        if layer_volumes.shape[vertical_axis] < 1:
            raise ValueError(
                "layer_volumes must contain at least one vertical layer."
            )
        surface_shape = (
            layer_volumes.shape[:vertical_axis]
            + layer_volumes.shape[vertical_axis + 1 :]
        )
        if tuple(flux.shape) != tuple(surface_shape):
            raise ValueError(
                "Surface flux shape must equal layer_volumes shape without "
                "the vertical axis."
            )
        if tuple(cell_area.shape) != tuple(surface_shape):
            raise ValueError(
                "cell_area shape must match the surface flux shape."
            )
        if not bool(torch.all(torch.isfinite(flux))):
            raise ValueError("Surface flux values must be finite.")
        if bool(torch.any(flux < 0)):
            raise ValueError("Surface emission flux values must be non-negative.")
        if (
            not bool(torch.all(torch.isfinite(cell_area)))
            or bool(torch.any(cell_area <= 0))
        ):
            raise ValueError("cell_area values must be finite and positive.")
        if (
            not bool(torch.all(torch.isfinite(layer_volumes)))
            or bool(torch.any(layer_volumes <= 0))
        ):
            raise ValueError(
                "layer_volumes values must be finite and positive."
            )

        constant_key = str(
            kwargs.get("molar_mass_g_per_mol_constant_key", "")
        ).strip()
        if not constant_key:
            raise RuntimeError(
                "SurfaceFluxToVolumeSourceAdapter requires "
                "molar_mass_g_per_mol_constant_key."
            )
        molar_mass_g_per_mol = float(get_physical_constant(constant_key))
        if not math.isfinite(molar_mass_g_per_mol) or molar_mass_g_per_mol <= 0:
            raise ValueError("Molar mass constant must be finite and positive.")
        molar_mass = torch.as_tensor(
            molar_mass_g_per_mol * 1.0e-3,
            dtype=flux.dtype,
            device=flux.device,
        )

        # References: Patankar (1980); Crank (1975); SI Brochure, 9th ed.
        # Expression: S_0 = F * M * A / V_0 and S_j = 0 for j > 0.
        # Parameters: F [mol m-2 s-1], M [kg mol-1], A [m2],
        # V [m3], and S [kg m-3 s-1].
        lower_volume = layer_volumes.select(vertical_axis, 0)
        column_mass_rate = flux * molar_mass * cell_area
        lower_source = column_mass_rate / lower_volume
        layer_selector = torch.zeros(
            layer_volumes.shape[vertical_axis],
            dtype=flux.dtype,
            device=flux.device,
        )
        layer_selector[0] = 1.0
        selector_shape = [1] * layer_volumes.ndim
        selector_shape[vertical_axis] = layer_selector.numel()
        source = (
            lower_source.unsqueeze(vertical_axis)
            * layer_selector.reshape(selector_shape)
        )

        reconstructed_rate = (source * layer_volumes).sum(
            dim=vertical_axis
        )
        column_residual = reconstructed_rate - column_mass_rate
        domain_input_rate = column_mass_rate.sum()
        domain_output_rate = reconstructed_rate.sum()
        domain_residual = domain_output_rate - domain_input_rate
        return AdapterResult(
            value=source,
            metadata={
                "adapter_type": self.adapter_type,
                "vertical_axis": vertical_axis,
                "injected_layer_index": 0,
                "molar_mass_constant_key": constant_key,
                "molar_mass_kg_per_mol": float(
                    molar_mass.detach().cpu().item()
                ),
                "column_input_mass_rates": self._to_flat_list(
                    column_mass_rate
                ),
                "column_output_mass_rates": self._to_flat_list(
                    reconstructed_rate
                ),
                "column_mass_rate_residuals": self._to_flat_list(
                    column_residual
                ),
                "domain_input_mass_rate": float(
                    domain_input_rate.detach().cpu().item()
                ),
                "domain_output_mass_rate": float(
                    domain_output_rate.detach().cpu().item()
                ),
                "domain_mass_rate_residual": float(
                    domain_residual.detach().cpu().item()
                ),
            },
        )


# Area measures of the PortSpec support vocabulary that refer a value to a unit of leaf or ground
# area. The area_basis of a support port is only the measure of its weights and values; the
# quantity basis itself comes from the SSOT catalog entry (spatial_basis) of the port variable.
_UNIT_AREA_MEASURES = frozenset({"leaf_area", "ground_area"})
# Inputs of the support reduction that only the governed ExecutableAdapterFunctor supplies.
_SUPPORT_GOVERNED_INPUTS = (
    "conservation_tolerance",
    "quantity_basis",
    "source_port_spec",
    "target_port_spec",
)


def _ssot_quantity_basis(ssot_catalog: Any, port: PortSpec, role: str) -> SpatialQuantityBasis:
    """The spatial_basis of a support port variable, read only from its SSOT catalog entry.

    The entry is looked up by the port name and, for a renamed port, by its standard_name.
    """
    entry = ssot_catalog.get(port.name)
    if entry is None and port.standard_name:
        entry = ssot_catalog.get(port.standard_name)
    if entry is None:
        raise ValueError(
            f"The {role} PortSpec {port.name!r} has no SSOT catalog entry; support aggregation "
            "takes the quantity basis only from the spatial_basis of that entry."
        )
    if entry.spatial_basis is None:
        raise ValueError(
            f"The SSOT entry {entry.name!r} of the {role} PortSpec {port.name!r} declares no "
            "spatial_basis (intensive, cell_total, area_density or categorical); support "
            "aggregation takes the quantity basis only from it."
        )
    return entry.spatial_basis


def _check_support_measure(
    basis: SpatialQuantityBasis,
    source_port: Any,
    target_port: Any,
) -> None:
    """The area_basis of the ports is the area measure of the reduction and must fit the basis."""
    for role, port in (("source", source_port), ("target", target_port)):
        if not isinstance(port, PortSpec):
            raise ValueError(
                f"Support aggregation needs the bound {role} PortSpec ({role}_port_spec); got "
                f"{type(port).__name__}."
            )
    if source_port.area_basis != target_port.area_basis:
        raise ValueError(
            f"The source PortSpec {source_port.name!r} declares area_basis "
            f"{source_port.area_basis!r} and the target PortSpec {target_port.name!r} "
            f"{target_port.area_basis!r}; a support reduction keeps the area measure (convert it "
            "with the area_basis_transform adapter first)."
        )
    measure = source_port.area_basis
    if basis is SpatialQuantityBasis.CELL_TOTAL and measure in _UNIT_AREA_MEASURES:
        raise ValueError(
            f"The support variable {source_port.name!r} is a cell total (an amount per cell) but "
            f"its ports declare the area measure {measure!r}; a cell total has no leaf or ground "
            "area measure."
        )
    if basis is SpatialQuantityBasis.AREA_DENSITY and measure is None:
        raise ValueError(
            f"The support variable {source_port.name!r} is an area density but its ports declare "
            "no area_basis; the area measure of the density (leaf_area, ground_area or "
            "grid_area) is required."
        )


class SupportAggregationAdapter(AdapterBase):
    """Aggregate a field to an explicit spatial support on its declared quantity basis."""

    adapter_type = "support_aggregation"

    def apply(self, value: Any, **kwargs: Any) -> AdapterResult:
        """
        References:
            Kljun et al. (2015), Geosci. Model Dev. 8, 3695-3713 (footprint-weighted
            flux); Jones (1999), conservative remapping; CF cell methods; the shared
            whole-support law and ledger aggregate_spatial_alias
            (relic/coupling/scales/alias_utils.py).
        Mathematical Expression:
            by the SSOT spatial_basis of the port variable: intensive and area_density
            y = sum_i w_i x_i / sum_i w_i (an area density stays an amount per unit of
            its area measure); cell_total Y = sum_i X_i; categorical the class with the
            largest weight share; ledger of an amount sum_i w_i x_i = y * sum_i w_i
            (area density) or sum_i X_i = Y (cell total).
        Parameters:
            value: source field x_i; weights: support measures w_i (for example
            normalized flux-footprint weights alpha_i); quantity_basis: the SSOT
            spatial_basis of the port variable (read at registration from the SSOT
            catalog of the NodeRegistry and bound through the NodeDefinition);
            source_port_spec and target_port_spec: the bound PortSpecs, whose area_basis
            is only the area measure; conservation_tolerance: parameter-backed relative
            ledger tolerance; strategy: optional ScalingStrategy value naming the basis
            law; enforce_conservation: raise (True) or report (False) a ledger residual.
        Units:
            x and y share the port units (for example mol m-2 s-1); w carries the
            support-measure units (1 for normalized weights); the ledger is x times w.
        Domain/constraints:
            both ports declare the same area_basis; a cell total has no leaf or ground
            area measure and takes no weights; an area density needs an area measure and
            weights; weights are finite, non-negative, of the source shape, with a
            positive total; a strategy outside the basis law raises; nothing is rescaled.
        Sign convention:
            the sign of x is preserved; weights are non-negative measures of support.
        """
        basis_value = kwargs.get("quantity_basis")
        if basis_value is None:
            raise ValueError(
                "Support aggregation needs quantity_basis: the SSOT spatial_basis of the port "
                "variable, bound by ExecutableAdapterFunctor from its NodeDefinition."
            )
        basis = SpatialQuantityBasis.parse(basis_value)
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        if not torch.is_floating_point(tensor) and basis is not SpatialQuantityBasis.CATEGORICAL:
            tensor = tensor.to(torch.float32)
        source_port = kwargs.get("source_port_spec")
        target_port = kwargs.get("target_port_spec")
        _check_support_measure(basis, source_port, target_port)
        tolerance = kwargs.get("conservation_tolerance")
        if tolerance is None:
            raise ValueError(
                "Support aggregation needs conservation_tolerance "
                "(coupling.scales.defaults.conservation_tolerance, supplied by "
                "ExecutableAdapterFunctor)."
            )
        enforce_conservation = kwargs.get("enforce_conservation", False)
        if not isinstance(enforce_conservation, bool):
            raise TypeError("Support aggregation enforce_conservation must be a bool.")
        strategy_name = kwargs.get("strategy")
        strategy: Optional[ScalingStrategy] = None
        if strategy_name is not None:
            try:
                strategy = ScalingStrategy(str(strategy_name).strip().lower())
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported support aggregation strategy {strategy_name!r}; a strategy "
                    f"must be a ScalingStrategy value {[item.value for item in ScalingStrategy]}."
                ) from exc

        raw_weights = kwargs.get("weights")
        weights: Optional[Tensor] = None
        if raw_weights is not None:
            if basis is SpatialQuantityBasis.CELL_TOTAL:
                raise ValueError(
                    "The support variable is a cell total: its amounts are summed over the whole "
                    "support and support weights do not apply; declare no weights_state."
                )
            weights = torch.as_tensor(
                raw_weights,
                dtype=tensor.dtype if torch.is_floating_point(tensor) else torch.float64,
                device=tensor.device,
            )

        variable_name = str(
            kwargs.get("variable_name", "support_value")
        ).strip()
        source_scale = str(kwargs.get("source_scale", "source")).strip()
        target_scale = str(kwargs.get("target_scale", "target")).strip()
        result, audit = aggregate_spatial_alias(
            data_dict={variable_name: tensor},
            variable_bases={variable_name: basis},
            source_scale=source_scale,
            target_scale=target_scale,
            mode=ScalingMode.CONSERVATIVE,
            device=tensor.device,
            enforce_conservation=enforce_conservation,
            tolerance=float(tolerance),
            weights=weights,
            strategy=strategy,
        )
        ledgers = {
            name: {
                "source_total": item.source_total,
                "target_total": item.target_total,
                "absolute_error": item.absolute_error,
                "relative_error": item.relative_error,
                "tolerance": item.tolerance,
                "is_conserved": item.is_conserved,
                "support_measure": item.metadata["support_measure"],
            }
            for name, item in audit.metadata["conservation_audit"].items()
        }
        return AdapterResult(
            value=result[variable_name],
            metadata={
                "adapter_type": self.adapter_type,
                "strategy": None if strategy is None else strategy.value,
                "quantity_basis": basis.value,
                "area_basis": source_port.area_basis,
                "source_scale": source_scale,
                "target_scale": target_scale,
                "conservation_enforced": audit.conservation_enforced,
                "mass_balance_error": audit.mass_balance_error,
                "relative_error": audit.relative_error,
                "conservation_audited_variables": list(
                    audit.metadata["conservation_audited_variables"]
                ),
                "conservation_audit": ledgers,
                "warnings": list(audit.warnings),
            },
        )


class ExecutableAdapterFunctor(ScaleCouplingFunctor):
    """Execute one registered edge adapter inside the Functor lifecycle."""

    def __init__(
        self,
        name: Optional[str] = None,
        adapter_type: str = "",
        source_node: str = "",
        source_port: str = "",
        target_port: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        adapter_type = str(adapter_type).strip()
        source_node = str(source_node).strip()
        source_port = str(source_port).strip()
        target_port = str(target_port).strip()
        if not all((adapter_type, source_node, source_port, target_port)):
            raise ValueError(
                "ExecutableAdapterFunctor requires adapter_type, source_node, "
                "source_port, and target_port."
            )
        if source_port == target_port:
            raise ValueError(
                "Executable adapter source_port and target_port must differ "
                "to preserve single-writer provenance."
            )

        adapter_config = dict(config or {})
        source_mode = adapter_config.get("source_mode", "subprocess_outputs")
        if not isinstance(source_mode, str) or source_mode not in (
            "subprocess_outputs", "state"
        ):
            raise ValueError(
                "ExecutableAdapterFunctor source_mode must be "
                "'subprocess_outputs' or 'state'."
            )
        if "dt" in adapter_config:
            raise ValueError(
                "Adapter config must not duplicate the simulator time step."
            )
        context_input_names = []
        for config_key in (
            "sample_durations_state",
            "leaf_area_index_state",
            "weights_state",
            "cell_area_state",
            "layer_volumes_state",
        ):
            state_name = adapter_config.get(config_key)
            if state_name is None:
                continue
            if not isinstance(state_name, str) or not state_name.strip():
                raise ValueError(
                    f"Adapter config '{config_key}' must be a non-empty "
                    "state variable name."
                )
            context_input_names.append(state_name.strip())
        if adapter_type == "support_aggregation":
            # The bound PortSpecs and the parameter-backed tolerance are governed inputs of the
            # support reduction; configuration cannot replace them.
            duplicated = sorted(
                set(adapter_config).intersection(_SUPPORT_GOVERNED_INPUTS)
            )
            if duplicated:
                raise ValueError(
                    f"Adapter config must not declare {duplicated}: the support adapter "
                    "receives the bound PortSpecs, the SSOT spatial_basis and the "
                    "conservation tolerance from its governed functor."
                )
        registry = AdapterRegistry()
        adapter = registry.get(adapter_type)
        if adapter is None:
            raise ValueError(f"Unknown adapter type: {adapter_type}")

        functor_name = name or f"{adapter_type}_adapter"
        metadata = FunctorMetadata(
            name=functor_name,
            version="1.0.0",
            description=(
                "Explicit graph-edge adapter delegated to the canonical "
                "AdapterRegistry."
            ),
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression=(
                "y = A(x; c), where A is the selected registered unit, "
                "temporal, semantic, area-basis, or support operator"
            ),
            references=[
                "CF Metadata Conventions 1.8, cell_methods",
                "Jones, P.W. (1999). Conservative remapping schemes.",
            ],
            parameters={
                "adapter_type": "Registered adapter selector [dimensionless]",
                "source_node": "Declared upstream graph node [dimensionless]",
                "source_port": "Declared upstream output port [dimensionless]",
                "target_port": "Declared transformed output port [dimensionless]",
                **(
                    {
                        "conservation_tolerance": (
                            "Relative ledger tolerance of the support reduction "
                            "from coupling.scales.defaults [dimensionless]"
                        )
                    }
                    if adapter_type == "support_aggregation"
                    else {}
                ),
            },
            input_variables=list(
                dict.fromkeys([source_port, *context_input_names])
            ),
            output_variables=[target_port],
            conservation_laws=[],
            spatial_scale=SpatialScale.UNKNOWN,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        super().__init__(
            name=functor_name,
            config=adapter_config,
            metadata=metadata,
            source_scale="declared_source_support",
            target_scale="declared_target_support",
            coupling_type="hierarchical",
            conservation_properties=[],
        )
        self.adapter_type = adapter_type
        self.source_mode = source_mode
        self.source_node = source_node
        self.source_port = source_port
        self.target_port = target_port
        self.adapter_config = adapter_config
        self._adapter_registry = registry
        self._adapter = adapter
        self._source_port_spec: Optional[PortSpec] = None
        self._target_port_spec: Optional[PortSpec] = None
        self.last_adapter_metadata: Dict[str, Any] = {}
        self._conservation_tolerance: Optional[float] = None
        self._support_quantity_basis: Optional[SpatialQuantityBasis] = None
        if adapter_type == "support_aggregation":
            tolerance = float(
                self.get_parameter_value(
                    "conservation_tolerance",
                    namespace=ParameterNamespace.COUPLING_SCALES_DEFAULTS.value,
                    units="dimensionless",
                    validate=True,
                    required=True,
                )
            )
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(
                    "coupling.scales.defaults.conservation_tolerance must be finite "
                    "and positive."
                )
            self._conservation_tolerance = tolerance

    def contract_config(self) -> Dict[str, Any]:
        """Return the serializable constructor contract for registry binding."""
        return {
            "adapter_type": self.adapter_type,
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_port": self.target_port,
            **deepcopy(self.adapter_config),
        }

    def bind_node_definition(self, definition: NodeDefinition) -> None:
        """Bind authoritative runtime port semantics from NodeRegistry."""
        source_spec = definition.inputs.get(self.source_port)
        target_spec = definition.outputs.get(self.target_port)
        if source_spec is None or target_spec is None:
            raise ValueError(
                "Executable adapter NodeDefinition does not contain its "
                "declared source and target ports."
            )
        self._source_port_spec = deepcopy(source_spec)
        self._target_port_spec = deepcopy(target_spec)
        if self.adapter_type == "support_aggregation":
            basis_value = definition.metadata.get("support_quantity_basis")
            if basis_value is None:
                raise ValueError(
                    "Executable support adapter NodeDefinition carries no "
                    "support_quantity_basis; register it through "
                    "AdapterRegistry.register_executable_node, which reads the "
                    "spatial_basis of the port variables from the SSOT catalog."
                )
            self._support_quantity_basis = SpatialQuantityBasis.parse(basis_value)

    def _target_write_mode(self) -> StateWriteMode:
        if self._target_port_spec is None:
            raise RuntimeError(
                "Executable adapter requires an authoritative NodeDefinition "
                "before execution."
            )
        raw_mode = self._target_port_spec.write_mode
        if raw_mode is None:
            return (
                StateWriteMode.DERIVED
                if str(self._target_port_spec.semantics).lower()
                == "diagnostic"
                else StateWriteMode.MUTABLE
            )
        try:
            return StateWriteMode(str(raw_mode).lower())
        except ValueError as exc:
            raise ValueError(
                f"Unsupported adapter target write_mode: {raw_mode}"
            ) from exc

    @staticmethod
    def _unwrap_update(value: Any) -> Any:
        return value.value if isinstance(value, UpdateItem) else value

    def _resolve_adapter_kwargs(
        self,
        *,
        state: Mapping[str, Any],
        dt: Any,
    ) -> Dict[str, Any]:
        adapter_kwargs = dict(self.adapter_config)
        if self.adapter_type == "support_aggregation":
            if (
                self._source_port_spec is None
                or self._target_port_spec is None
                or self._support_quantity_basis is None
            ):
                raise RuntimeError(
                    "Executable support adapter requires an authoritative "
                    "NodeDefinition (bound source and target PortSpecs and the SSOT "
                    "spatial_basis) before execution."
                )
            adapter_kwargs["quantity_basis"] = self._support_quantity_basis
            adapter_kwargs["source_port_spec"] = self._source_port_spec
            adapter_kwargs["target_port_spec"] = self._target_port_spec
            adapter_kwargs["conservation_tolerance"] = self._conservation_tolerance
        if self.adapter_type in {"semantic_adapter", "flux_to_increment"}:
            adapter_kwargs["dt"] = dt
        if (
            self.adapter_type == "temporal_resample"
            and adapter_kwargs.get("sampling_mode") == "current_step"
        ):
            adapter_kwargs["dt"] = dt
        duration_key = adapter_kwargs.pop("sample_durations_state", None)
        if duration_key is not None:
            if duration_key not in state:
                raise KeyError(
                    f"Adapter sample duration state '{duration_key}' is missing."
                )
            adapter_kwargs["sample_durations"] = state[duration_key]
        lai_key = adapter_kwargs.pop("leaf_area_index_state", None)
        if lai_key is not None:
            if lai_key not in state:
                raise KeyError(
                    f"Adapter leaf area index state '{lai_key}' is missing."
                )
            adapter_kwargs["leaf_area_index"] = state[lai_key]
        weights_key = adapter_kwargs.pop("weights_state", None)
        if weights_key is not None:
            if weights_key not in state:
                raise KeyError(
                    f"Adapter support weights state '{weights_key}' is missing."
                )
            adapter_kwargs["weights"] = state[weights_key]
        cell_area_key = adapter_kwargs.pop("cell_area_state", None)
        if cell_area_key is not None:
            if cell_area_key not in state:
                raise KeyError(
                    f"Adapter cell area state '{cell_area_key}' is missing."
                )
            adapter_kwargs["cell_area"] = state[cell_area_key]
        layer_volumes_key = adapter_kwargs.pop(
            "layer_volumes_state",
            None,
        )
        if layer_volumes_key is not None:
            if layer_volumes_key not in state:
                raise KeyError(
                    "Adapter layer volumes state "
                    f"'{layer_volumes_key}' is missing."
                )
            adapter_kwargs["layer_volumes"] = state[layer_volumes_key]
        return adapter_kwargs

    def forward(
        self,
        state: Dict[str, Tensor],
        time: float,
        dt: Any,
        subprocess_outputs: Dict[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, UpdateItem]:
        """Execute the registered transformation from its explicitly selected input."""
        # References: CF Metadata Conventions 1.8; Jones (1999);
        #   RELIC architecture Sections 3 and 5, declared input projection.
        # Mathematical expression: x = state[s] in explicit state mode;
        #   otherwise x = P[n][s]; y = A(x; c), with the registered operator A.
        # Parameter meanings: s is the declared source port; n is the upstream
        #   node; P is its dependency-output map; c is declared adapter context.
        # Units: x and y retain their bound source/target PortSpec units;
        #   dt [s]; source_mode is a dimensionless input-channel selector.
        # Domain/constraints: only a declared state source or upstream output
        #   is read; missing inputs raise; selection never falls back.
        # Sign convention: signs and flux directions follow A's existing law;
        #   this binding adds no transfer and does not advance state.
        if self.source_mode == "state":
            if self.source_port not in state:
                raise KeyError(
                    f"Adapter state source port '{self.source_port}' is unavailable."
                )
            source_value = state[self.source_port]
        else:
            upstream = subprocess_outputs.get(self.source_node)
            if not isinstance(upstream, dict):
                raise RuntimeError(
                    f"Adapter upstream output '{self.source_node}' is unavailable."
                )
            if self.source_port not in upstream:
                raise KeyError(
                    f"Adapter source port '{self.source_port}' is unavailable "
                    f"from '{self.source_node}'."
                )
            source_value = upstream[self.source_port]
        value = self._unwrap_update(source_value)
        adapter_kwargs = self._resolve_adapter_kwargs(state=state, dt=dt)
        adapter_kwargs.pop("source_mode", None)
        result = self._adapter.apply(value, **adapter_kwargs)
        self.last_adapter_metadata = dict(result.metadata)
        if self._target_port_spec is None:
            raise RuntimeError(
                "Executable adapter target PortSpec is not bound."
            )
        return {
            self.target_port: UpdateItem(
                value=result.value,
                mode=UpdateMode.REPLACE,
                write_mode=self._target_write_mode(),
                unit=self._target_port_spec.units,
                per_timestep=(
                    self._target_port_spec.per_timestep
                    or self.adapter_type
                    in {"semantic_adapter", "flux_to_increment"}
                ),
                source_functor=self.name,
            )
        }


class AdapterRegistry:
    """Registry and factory for adapters."""

    def __init__(self, unit_converter: Optional[UnitConverter] = None) -> None:
        self.unit_converter = (
            unit_converter
            if unit_converter is not None
            else canonical_unit_converter
        )
        semantic_adapter = SemanticAdapter()
        self._adapters = {
            "unit_cast": UnitCastAdapter(self.unit_converter),
            "temporal_resample": TemporalResampleAdapter(),
            "semantic_adapter": semantic_adapter,
            "flux_to_increment": semantic_adapter,
            "area_basis_transform": AreaBasisTransformAdapter(),
            "surface_flux_to_volume_source": (
                SurfaceFluxToVolumeSourceAdapter()
            ),
            "support_aggregation": SupportAggregationAdapter(),
        }

    def get(self, adapter_type: str) -> Optional[AdapterBase]:
        return self._adapters.get(adapter_type)

    @staticmethod
    def _shape_dimensions(port: PortSpec, label: str) -> list[str]:
        shape = port.shape
        dimensions = (
            shape.get("dimensions")
            if isinstance(shape, Mapping)
            else None
        )
        if (
            not isinstance(dimensions, (list, tuple))
            or any(
                not isinstance(dimension, str) or not dimension.strip()
                for dimension in dimensions
            )
        ):
            raise ValueError(
                f"{label} must declare non-empty shape dimension names."
            )
        return [dimension.strip() for dimension in dimensions]

    def _validate_executable_shape_contract(
        self,
        *,
        functor: ExecutableAdapterFunctor,
        source_port: PortSpec,
        target_port: PortSpec,
        context_ports: Mapping[str, PortSpec],
    ) -> None:
        if functor.adapter_type == "temporal_resample":
            source_dimensions = self._shape_dimensions(
                source_port,
                "Temporal adapter source PortSpec",
            )
            target_dimensions = self._shape_dimensions(
                target_port,
                "Temporal adapter target PortSpec",
            )
            sampling_mode = str(
                functor.adapter_config.get("sampling_mode", "window")
            ).strip().lower()
            if sampling_mode == "current_step":
                if functor.adapter_config.get("method") != "interval_mean":
                    raise ValueError(
                        "Current-step temporal adapters require "
                        "method='interval_mean'."
                    )
                if "axis" in functor.adapter_config:
                    raise ValueError(
                        "Current-step temporal adapters must not declare axis."
                    )
                if source_dimensions != target_dimensions:
                    raise ValueError(
                        "Current-step temporal adapters require identical "
                        "source and target shapes."
                    )
                if source_port.temporal_basis != "instantaneous":
                    raise ValueError(
                        "Current-step temporal adapters require an "
                        "instantaneous source temporal basis."
                    )
                if target_port.temporal_basis != "interval_mean":
                    raise ValueError(
                        "Current-step temporal adapters require an "
                        "interval_mean target temporal basis."
                    )
                duration_state = functor.adapter_config.get(
                    "sample_durations_state"
                )
                if (
                    not isinstance(duration_state, str)
                    or not duration_state.strip()
                    or duration_state not in context_ports
                ):
                    raise ValueError(
                        "Current-step temporal adapters require an explicit "
                        "sample_durations_state PortSpec."
                    )
                duration_dimensions = self._shape_dimensions(
                    context_ports[duration_state],
                    "Current-step observation-duration PortSpec",
                )
                if duration_dimensions:
                    raise ValueError(
                        "Current-step observation-duration PortSpec must be "
                        "scalar."
                    )
                return
            if sampling_mode != "window":
                raise ValueError(
                    "Temporal adapter sampling_mode must be 'window' or "
                    "'current_step'."
                )
            axis = int(functor.adapter_config.get("axis", 0))
            if axis < 0:
                axis += len(source_dimensions)
            if axis < 0 or axis >= len(source_dimensions):
                raise ValueError(
                    "Temporal adapter axis is outside the declared source shape."
                )
            temporal_axis = source_dimensions[axis]
            if temporal_axis not in {"sample", "time", "time_window", "window"}:
                raise ValueError(
                    "Temporal adapter source must reduce a named temporal axis."
                )
            expected_target = [
                dimension
                for index, dimension in enumerate(source_dimensions)
                if index != axis
            ]
            if target_dimensions != expected_target:
                raise ValueError(
                    "Temporal adapter target shape must remove exactly the "
                    "declared temporal source axis."
                )
            duration_state = functor.adapter_config.get(
                "sample_durations_state"
            )
            if isinstance(duration_state, str):
                duration_dimensions = self._shape_dimensions(
                    context_ports[duration_state],
                    "Temporal sample-duration PortSpec",
                )
                if duration_dimensions != [temporal_axis]:
                    raise ValueError(
                        "Temporal sample-duration shape must match the reduced "
                        "temporal axis."
                    )
            return

        if functor.adapter_type == "support_aggregation":
            source_dimensions = self._shape_dimensions(
                source_port,
                "Support adapter source PortSpec",
            )
            target_dimensions = self._shape_dimensions(
                target_port,
                "Support adapter target PortSpec",
            )
            if target_dimensions:
                raise ValueError(
                    "Support adapter target must declare a scalar shape."
                )
            weights_state = functor.adapter_config.get("weights_state")
            if isinstance(weights_state, str):
                weight_dimensions = self._shape_dimensions(
                    context_ports[weights_state],
                    "Support adapter weights PortSpec",
                )
                if weight_dimensions != source_dimensions:
                    raise ValueError(
                        "Support adapter weights shape must match the source "
                        "shape."
                    )
            return

        if functor.adapter_type == "surface_flux_to_volume_source":
            source_dimensions = self._shape_dimensions(
                source_port,
                "Surface flux adapter source PortSpec",
            )
            target_dimensions = self._shape_dimensions(
                target_port,
                "Surface flux adapter target PortSpec",
            )
            vertical_axis = int(
                functor.adapter_config.get("vertical_axis", -1)
            )
            if vertical_axis < 0:
                vertical_axis += len(target_dimensions)
            if vertical_axis < 0 or vertical_axis >= len(target_dimensions):
                raise ValueError(
                    "Surface flux adapter vertical_axis is outside the "
                    "target shape."
                )
            expected_surface_dimensions = [
                dimension
                for index, dimension in enumerate(target_dimensions)
                if index != vertical_axis
            ]
            if source_dimensions != expected_surface_dimensions:
                raise ValueError(
                    "Surface flux adapter target shape must add exactly one "
                    "declared vertical axis."
                )
            area_state = functor.adapter_config.get("cell_area_state")
            volume_state = functor.adapter_config.get(
                "layer_volumes_state"
            )
            if not isinstance(area_state, str) or area_state not in context_ports:
                raise ValueError(
                    "Surface flux adapter requires cell_area_state."
                )
            if (
                not isinstance(volume_state, str)
                or volume_state not in context_ports
            ):
                raise ValueError(
                    "Surface flux adapter requires layer_volumes_state."
                )
            area_port = context_ports[area_state]
            volume_port = context_ports[volume_state]
            if self._shape_dimensions(
                area_port,
                "Surface flux adapter cell-area PortSpec",
            ) != source_dimensions:
                raise ValueError(
                    "Surface flux adapter cell-area shape must match the "
                    "source shape."
                )
            if self._shape_dimensions(
                volume_port,
                "Surface flux adapter layer-volume PortSpec",
            ) != target_dimensions:
                raise ValueError(
                    "Surface flux adapter layer-volume shape must match the "
                    "target shape."
                )
            expected_units = {
                "source": (source_port.units, "mol m-2 s-1"),
                "target": (target_port.units, "kg m-3 s-1"),
                "cell_area": (area_port.units, "m2"),
                "layer_volumes": (volume_port.units, "m3"),
            }
            for label, (actual, expected) in expected_units.items():
                if actual != expected:
                    raise ValueError(
                        f"Surface flux adapter {label} units must be "
                        f"'{expected}'."
                    )
            constant_key = functor.adapter_config.get(
                "molar_mass_g_per_mol_constant_key"
            )
            if not isinstance(constant_key, str) or not constant_key.strip():
                raise ValueError(
                    "Surface flux adapter requires a molar mass physical "
                    "constant key."
                )
            get_physical_constant(constant_key.strip())

    def _validate_executable_unit_contract(
        self,
        *,
        functor: ExecutableAdapterFunctor,
        source_port: PortSpec,
        target_port: PortSpec,
    ) -> None:
        """
        Validate one unit_cast against the canonical conversion owner.

        References:
            CF Metadata Conventions 1.8 unit declarations.
        Mathematical Expression:
            x_target = U(x_source; variable_name, source_units, target_units).
        Parameters:
            functor: executable adapter declaration.
            source_port: authoritative input unit and standard name.
            target_port: authoritative output unit and standard name.
        """
        if functor.adapter_type != "unit_cast":
            return
        variable_name = str(functor.adapter_config.get("variable_name", "")).strip()
        source_units = str(functor.adapter_config.get("source_units", "")).strip()
        target_units = str(functor.adapter_config.get("target_units", "")).strip()
        if not all((variable_name, source_units, target_units)):
            raise ValueError(
                "Executable unit_cast requires variable_name, source_units, "
                "and target_units."
            )
        if (
            source_port.standard_name != variable_name
            or target_port.standard_name != variable_name
        ):
            raise ValueError(
                "Executable unit_cast variable_name must match both PortSpec "
                "standard names."
            )
        if source_port.units != source_units or target_port.units != target_units:
            raise ValueError(
                "Executable unit_cast units must exactly match the source and "
                "target PortSpec units."
            )
        try:
            conversion_supported = self.unit_converter.can_convert(
                source_units,
                target_units,
                variable_name,
            )
            if conversion_supported:
                self.unit_converter.convert(
                    1.0,
                    variable_name=variable_name,
                    source_units=source_units,
                    target_units=target_units,
                    context=None,
                )
        except (TypeError, ValueError, RuntimeError):
            conversion_supported = False
        if not conversion_supported:
            raise ValueError(
                "Executable unit_cast is not executable through the canonical "
                "UnitConverter without an undeclared conversion context."
            )

    def register_executable_node(
        self,
        *,
        node_registry: "NodeRegistry",
        node_type_id: str,
        functor: ExecutableAdapterFunctor,
        source_port: PortSpec,
        target_port: PortSpec,
        context_ports: Optional[Dict[str, PortSpec]] = None,
    ) -> NodeDefinition:
        """Bind one adapter alias to real runtime and authoritative ports."""
        if self.get(functor.adapter_type) is None:
            raise ValueError(
                f"Adapter type '{functor.adapter_type}' is not registered."
            )
        if source_port.name != functor.source_port:
            raise ValueError(
                "Adapter source PortSpec name does not match the functor "
                "source_port."
            )
        if target_port.name != functor.target_port:
            raise ValueError(
                "Adapter target PortSpec name does not match the functor "
                "target_port."
            )
        expected_context_names = set(functor.metadata.input_variables) - {
            functor.source_port
        }
        provided_context_ports = dict(context_ports or {})
        if set(provided_context_ports) != expected_context_names:
            raise ValueError(
                "Executable adapter context PortSpec names must exactly "
                f"match {sorted(expected_context_names)}."
            )
        for context_name, context_port in provided_context_ports.items():
            if context_port.name != context_name:
                raise ValueError(
                    "Adapter context PortSpec key and name must match."
                )
        self._validate_executable_unit_contract(
            functor=functor,
            source_port=source_port,
            target_port=target_port,
        )
        self._validate_executable_shape_contract(
            functor=functor,
            source_port=source_port,
            target_port=target_port,
            context_ports=provided_context_ports,
        )
        support_basis: Optional[SpatialQuantityBasis] = None
        if functor.adapter_type == "support_aggregation":
            support_basis = self._support_basis_contract(
                node_registry=node_registry,
                functor=functor,
                source_port=source_port,
                target_port=target_port,
            )
        definition = node_registry.register_functor_instance(
            functor,
            node_type_id=node_type_id,
        )
        copied_source = deepcopy(source_port)
        copied_source.direction = "in"
        copied_source.write_mode = None
        copied_source.required = True
        definition.inputs = {source_port.name: copied_source}
        for context_name, context_port in provided_context_ports.items():
            copied_port = deepcopy(context_port)
            copied_port.direction = "in"
            copied_port.write_mode = None
            copied_port.required = True
            definition.inputs[context_name] = copied_port
        copied_target = deepcopy(target_port)
        copied_target.direction = "out"
        definition.outputs = {target_port.name: copied_target}
        definition.metadata["adapter_contract"] = functor.contract_config()
        if support_basis is not None:
            # The SSOT spatial_basis travels with the NodeDefinition to the bound functor.
            definition.metadata["support_quantity_basis"] = support_basis.value
        return definition

    @staticmethod
    def _support_basis_contract(
        *,
        node_registry: "NodeRegistry",
        functor: ExecutableAdapterFunctor,
        source_port: PortSpec,
        target_port: PortSpec,
    ) -> SpatialQuantityBasis:
        """The SSOT spatial_basis of the support ports, checked against their measure and weights."""
        source_basis = _ssot_quantity_basis(node_registry.ssot_catalog, source_port, "source")
        target_basis = _ssot_quantity_basis(node_registry.ssot_catalog, target_port, "target")
        if source_basis is not target_basis:
            raise ValueError(
                "The source and target SSOT entries of a support reduction must declare the "
                f"same spatial_basis; got {source_basis.value!r} for {source_port.name!r} and "
                f"{target_basis.value!r} for {target_port.name!r} (a support reduction keeps "
                "the basis)."
            )
        _check_support_measure(source_basis, source_port, target_port)
        weighted = functor.adapter_config.get("weights_state") is not None
        if source_basis is SpatialQuantityBasis.CELL_TOTAL and weighted:
            raise ValueError(
                f"The support variable {source_port.name!r} is a cell total: its amounts are "
                "summed over the whole support and support weights do not apply; declare no "
                "weights_state."
            )
        if source_basis is SpatialQuantityBasis.AREA_DENSITY and not weighted:
            raise ValueError(
                f"The support variable {source_port.name!r} is an area density: its support "
                "mean needs the support weights (weights_state)."
            )
        return source_basis

    def suggest_unit_adapter(self, source: PortSpec, target: PortSpec) -> Optional[str]:
        if not source.units or not target.units:
            return None
        if source.units == target.units:
            return None
        if self.unit_converter.can_convert(source.units, target.units, source.standard_name):
            return "unit_cast"
        if self.unit_converter.can_convert(source.units, target.units, "generic"):
            return "unit_cast"
        return None
