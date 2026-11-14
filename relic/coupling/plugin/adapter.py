"""Functor adapter that exposes external C plugins as RELIC functors."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ...core.functor.scale_coupling_bridge import ScaleCouplingFunctor
from ...core.functor.functor_types import (
    ExecutionPhase,
    FunctorMetadata,
    ProcessCategory,
    SpatialScale,
    VariableKind,
)
from ...core.state.types import UpdateMode
from ..contracts.manifest_validator import ManifestValidator
from .host import PluginHost


class PluginFunctorAdapter(ScaleCouplingFunctor):
    """Wrap a C ABI plugin as a standard RELIC Functor."""

    _TYPE_ID_TO_DTYPE: Dict[str, type] = {
        "INT32": np.int32,
        "INT64": np.int64,
        "F32": np.float32,
        "F64": np.float64,
        "STR": np.uint8,
        "BYTES": np.uint8,
    }

    def __init__(
        self,
        name: str,
        plugin_path: str,
        config: Optional[Dict[str, Any]] = None,
        host: Optional[PluginHost] = None,
        metadata: Optional[FunctorMetadata] = None,
        **kwargs: Any,
    ) -> None:
        self.plugin_path = str(plugin_path)
        self.host = host or PluginHost()
        self._manifest_validator = ManifestValidator()
        self._agreed_manifest: Dict[str, Any] = {}
        self._ports: List[Dict[str, Any]] = []

        base_metadata = metadata or FunctorMetadata(
            name=name,
            version="1.0.0",
            description="Language-agnostic plugin adapter functor",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression=(
                "STATE: x_new = value; INCREMENT: x_new = x_old + delta; "
                "FLUX: x_new = x_old + flux*dt when per_timestep is false."
            ),
            references=[
                "ESMF 8.9.1 NUOPC Reference Documentation",
                "LeVeque (2002) Finite Volume Methods",
            ],
            parameters={
                "plugin.path": "Path to plugin shared library [string]",
                "plugin.ports": "Port schema from negotiated plugin manifest [dimensionless]",
                "plugin.metadata.spatial_scale": "Required spatial scale declaration when manifest inference is ambiguous [enum]",
            },
            input_variables=["plugin_input_placeholder"],
            output_variables=["plugin_output_placeholder"],
            spatial_scale=SpatialScale.GRID_CELL,
            variable_kind=VariableKind.DIAGNOSTIC,
        )
        # A plugin couples the scales its host model declares, so the scales are required keys
        # of the instance configuration; none is assumed.
        adapter_config = dict(config or {})
        missing_scales = [
            key for key in ("source_scale", "target_scale", "coupling_type") if key not in adapter_config
        ]
        if missing_scales:
            raise KeyError(
                f"Plugin adapter '{name}' config declares no {missing_scales}; a plugin coupling "
                "states the scales it couples (ScaleCoupling source_scale, target_scale and "
                "coupling_type)."
            )
        super().__init__(
            name=name,
            config=adapter_config,
            metadata=base_metadata,
            source_scale=adapter_config["source_scale"],
            target_scale=adapter_config["target_scale"],
            coupling_type=adapter_config["coupling_type"],
            **kwargs,
        )

    def initialize(self, host_capabilities: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load plugin, negotiate manifest, and refresh functor metadata."""
        if self._ports:
            return dict(self._agreed_manifest)

        # A reloaded host is already bound to this path; a retired host must
        # be reloaded explicitly before it can serve a new manifest.
        if self.host.state == "retired":
            raise RuntimeError(
                f"Plugin adapter '{self.name}' is retired; reload it before initializing."
            )
        if not self.host.loaded:
            self.host.load(self.plugin_path)
        probed_manifest = self.host.probe()
        probe_errors = self._manifest_validator.validate(probed_manifest)
        if probe_errors:
            raise ValueError("Plugin probe manifest invalid: " + "; ".join(probe_errors))

        capabilities = {"host": "RELIC", "zero_copy": True, "interface_version": "1.0"}
        if host_capabilities:
            capabilities.update(host_capabilities)
        agreed_manifest = self.host.negotiate(capabilities)

        errors = self._manifest_validator.validate(agreed_manifest)
        if errors:
            raise ValueError("Plugin agreed manifest invalid: " + "; ".join(errors))

        ports = agreed_manifest.get("ports", [])
        if not isinstance(ports, list):
            raise ValueError("Plugin manifest field 'ports' must be a list.")
        self._ports = [dict(port) for port in ports if isinstance(port, dict)]
        self._agreed_manifest = dict(agreed_manifest)
        self.metadata = self._build_metadata_from_manifest(self._agreed_manifest)
        return dict(self._agreed_manifest)

    # ------------------------------------------------------------------
    # Native lifecycle (E03): the adapter never releases the library while
    # a native call is in flight; the host owns the drain boundary.
    # ------------------------------------------------------------------
    def retire(self) -> Dict[str, Any]:
        """Refuse new native calls through this adapter."""
        return self.host.retire()

    def unload(self, *, drain_timeout: Optional[float] = None) -> Dict[str, Any]:
        """Retire, drain, and release the plugin library (receipt returned)."""
        return self.host.unload(drain_timeout=drain_timeout)

    def reload(self) -> Dict[str, Any]:
        """Release and bind the same library again; the manifest is re-negotiated."""
        receipt = self.host.reload()
        self._ports = []
        self._agreed_manifest = {}
        return receipt

    @property
    def native_abi(self) -> Dict[str, Any]:
        """ABI record captured by the host when the library was loaded."""
        return self.host.abi

    def forward(
        self,
        state: Dict[str, Any],
        time: float,
        dt: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # References: ESMF 8.9.1 NUOPC Reference Documentation; LeVeque (2002); this
        #   adapter's FunctorMetadata.mathematical_expression; ManifestValidator contract.
        # Mathematical expression: per negotiated port x_out = plugin(x_in, t);
        #   update_mode STATE -> REPLACE (x_new = value), INCREMENT -> ACCUMULATE
        #   (x_new = x_old + delta), FLUX -> FLUX (x_new = x_old + flux dt when
        #   per_timestep is false); StateManager performs the advance, not this method.
        # Parameter meanings: port name, direction (in/out/inout), type_id, shape_hint,
        #   update_mode, per_timestep and units come from the manifest; t is the float
        #   timestamp handed to the plugin; dt is not used here.
        # Units: carried unchanged; UpdateItem.unit is the port "units" string when
        #   declared, otherwise None; no unit conversion is performed.
        # Domain/constraints: host not retired (RuntimeError); manifest negotiated on
        #   first call; each in/inout port must exist in state (KeyError); unnamed or
        #   unknown-direction ports skipped; C-contiguous buffers of port dtype (F64 default).
        # Sign convention: not applicable: values pass through from the plugin; any flux
        #   direction is defined by the plugin manifest, not by this adapter.
        if self.host.state == "retired":
            raise RuntimeError(
                f"Plugin adapter '{self.name}' is retired; no new native calls are accepted."
            )
        if not self._ports:
            self.initialize(host_capabilities=kwargs.get("host_capabilities"))

        outputs: Dict[str, Any] = {}
        timestamp = float(time)

        for port in self._ports:
            port_name = str(port.get("name", "")).strip()
            if not port_name:
                continue
            direction = str(port.get("direction", "inout")).strip().lower()
            if direction not in {"in", "out", "inout"}:
                continue

            if direction in {"in", "inout"}:
                if port_name not in state:
                    raise KeyError(f"Missing required plugin input port: {port_name}")
                source_value = state[port_name]
            else:
                source_value = 0.0
            buffer = self._to_numpy_buffer(source_value, port)
            self.host.exchange(port_name, buffer, timestamp)

            if direction not in {"out", "inout"}:
                continue

            resolved_value = self._to_state_value(buffer, state.get(port_name))
            mode = self._map_update_mode(port.get("update_mode"))
            item = self.wrap_update(
                resolved_value,
                mode=mode,
                per_timestep=bool(port.get("per_timestep", False)),
            )
            units = port.get("units")
            if units is not None:
                item.unit = str(units)
            outputs[port_name] = item

        return outputs

    def _build_metadata_from_manifest(self, manifest: Dict[str, Any]) -> FunctorMetadata:
        ports = manifest.get("ports", [])
        input_variables: List[str] = []
        output_variables: List[str] = []
        variable_kind = VariableKind.UNKNOWN

        for port in ports if isinstance(ports, list) else []:
            if not isinstance(port, dict):
                continue
            name = str(port.get("name", "")).strip()
            if not name:
                continue
            direction = str(port.get("direction", "inout")).strip().lower()
            update_mode = str(port.get("update_mode", "")).strip().upper()
            if direction in {"in", "inout"}:
                input_variables.append(name)
            if direction in {"out", "inout"}:
                output_variables.append(name)
            if update_mode == "FLUX":
                variable_kind = VariableKind.FLUX
            elif update_mode in {"STATE", "INCREMENT"} and variable_kind != VariableKind.FLUX:
                variable_kind = VariableKind.STATE

        inferred_scale = self._resolve_spatial_scale(manifest, ports)
        if inferred_scale == SpatialScale.UNKNOWN:
            raise ValueError(
                "Plugin manifest did not declare a resolvable spatial_scale. "
                "Provide config['spatial_scale'] or manifest metadata.spatial_scale."
            )
        if variable_kind == VariableKind.UNKNOWN:
            variable_kind = VariableKind.DIAGNOSTIC

        return FunctorMetadata(
            name=self.name,
            version="1.0.0",
            description=f"Plugin adapter for {manifest.get('name', self.name)}",
            category=ProcessCategory.SCALE_COUPLING,
            execution_phase=ExecutionPhase.COUPLING,
            mathematical_expression=(
                "STATE: x_new = value; INCREMENT: x_new = x_old + delta; "
                "FLUX: x_new = x_old + flux*dt when per_timestep is false."
            ),
            references=[
                "ESMF 8.9.1 NUOPC Reference Documentation",
                "LeVeque (2002) Finite Volume Methods",
            ],
            parameters={
                "plugin.path": "Path to plugin shared library [string]",
                "plugin.ports": "Port schema from negotiated plugin manifest [dimensionless]",
                "plugin.metadata.spatial_scale": "Plugin output spatial scale declaration [enum]",
            },
            input_variables=input_variables,
            output_variables=output_variables,
            spatial_scale=inferred_scale,
            variable_kind=variable_kind,
        )

    @classmethod
    def _infer_spatial_scale_from_shape_hint(cls, shape_hint: Any) -> SpatialScale:
        shape = cls._shape_from_hint(shape_hint)
        if len(shape) >= 3:
            return SpatialScale.GRID_3D
        if len(shape) >= 2:
            return SpatialScale.GRID_2D
        if len(shape) == 1:
            return SpatialScale.POINT
        return SpatialScale.UNKNOWN

    def _resolve_spatial_scale(self, manifest: Dict[str, Any], ports: Any) -> SpatialScale:
        configured_scale = None
        if isinstance(self.config, dict):
            configured_scale = self.config.get("spatial_scale")
        metadata_block = manifest.get("metadata", {})
        declared_scale = None
        if isinstance(metadata_block, dict):
            declared_scale = metadata_block.get("spatial_scale")
        if declared_scale is None:
            declared_scale = manifest.get("spatial_scale")
        if declared_scale is None:
            declared_scale = configured_scale
        scale = SpatialScale.from_value(declared_scale)
        if scale != SpatialScale.UNKNOWN:
            return scale

        inferred_scales: List[SpatialScale] = []
        for port in ports if isinstance(ports, list) else []:
            if not isinstance(port, dict):
                continue
            per_port_scale = SpatialScale.from_value(port.get("spatial_scale"))
            if per_port_scale != SpatialScale.UNKNOWN:
                inferred_scales.append(per_port_scale)
                continue
            inferred = self._infer_spatial_scale_from_shape_hint(port.get("shape_hint"))
            if inferred != SpatialScale.UNKNOWN:
                inferred_scales.append(inferred)

        if not inferred_scales:
            return SpatialScale.UNKNOWN

        priority = {
            SpatialScale.GRID_3D: 5,
            SpatialScale.GRID_2D: 4,
            SpatialScale.GRID_CELL: 3,
            SpatialScale.POINT: 2,
            SpatialScale.PATCH: 1,
        }
        return max(inferred_scales, key=lambda item: priority.get(item, 0))

    @classmethod
    def _map_update_mode(cls, raw_mode: Any) -> UpdateMode:
        mode = str(raw_mode or "INCREMENT").strip().upper()
        if mode == "STATE":
            return UpdateMode.REPLACE
        if mode == "FLUX":
            return UpdateMode.FLUX
        return UpdateMode.ACCUMULATE

    @classmethod
    def _dtype_for_port(cls, port: Dict[str, Any]) -> np.dtype:
        type_id = str(port.get("type_id", "F64")).strip().upper()
        return np.dtype(cls._TYPE_ID_TO_DTYPE.get(type_id, np.float64))

    @staticmethod
    def _shape_from_hint(shape_hint: Any) -> Tuple[int, ...]:
        if not isinstance(shape_hint, Sequence):
            return (1,)
        dims: List[int] = []
        for dim in shape_hint:
            try:
                value = int(dim)
            except (TypeError, ValueError):
                return (1,)
            if value <= 0:
                return (1,)
            dims.append(value)
        return tuple(dims) if dims else (1,)

    def _to_numpy_buffer(self, value: Any, port: Dict[str, Any]) -> np.ndarray:
        dtype = self._dtype_for_port(port)
        if isinstance(value, Tensor):
            array = value.detach().cpu().numpy().astype(dtype, copy=False)
        elif isinstance(value, np.ndarray):
            array = value.astype(dtype, copy=False)
        elif isinstance(value, (int, float, np.integer, np.floating)):
            array = np.asarray([value], dtype=dtype)
        else:
            shape = self._shape_from_hint(port.get("shape_hint"))
            array = np.zeros(shape, dtype=dtype)

        if array.ndim == 0:
            array = array.reshape(1)
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)
        return array

    @staticmethod
    def _to_state_value(buffer: np.ndarray, reference_value: Any) -> Any:
        if isinstance(reference_value, Tensor):
            target = torch.from_numpy(buffer.copy()).to(dtype=reference_value.dtype)
            if reference_value.numel() == target.numel():
                target = target.reshape(reference_value.shape)
            return target.to(reference_value.device)

        if isinstance(reference_value, np.ndarray):
            target = buffer.astype(reference_value.dtype, copy=False)
            if reference_value.shape and target.size == reference_value.size:
                return target.reshape(reference_value.shape)
            return target

        if buffer.size == 1:
            scalar = buffer.reshape(-1)[0]
            if np.issubdtype(buffer.dtype, np.integer):
                return int(scalar)
            return float(scalar)

        return torch.from_numpy(buffer.copy())


__all__ = ["PluginFunctorAdapter"]
