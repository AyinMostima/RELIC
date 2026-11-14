"""
Node registry for Graph IR orchestration.

Builds typed port definitions from SSOT catalogs and functor metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from copy import deepcopy
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional

from ..core.variable_mapper import variable_name_mapper
from .port_spec import PORT_SUPPORT_FIELDS, NodeDefinition, PortSpec
from .ssot_catalog import SSOTCatalog, SSOTEntry


def _normalize_names(names: Any) -> List[str]:
    if names is None:
        return []
    if isinstance(names, dict):
        return [str(name) for name in names.keys()]
    if isinstance(names, (list, tuple, set)):
        return [str(name) for name in names]
    return [str(names)]


@dataclass
class NodeRegistry:
    """Registry of node type definitions."""

    ssot_catalog: SSOTCatalog
    definitions: Dict[str, NodeDefinition] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # _build_port resolves every port name through this alias map; a catalog that
        # cannot be loaded raises here (its error names the alias or variable) instead of
        # leaving a stale or empty map behind.
        variable_name_mapper.refresh_from_parameter_manager()

    def register_definition(self, node_type_id: str, definition: NodeDefinition) -> None:
        self.definitions[node_type_id] = definition

    def unregister_definition(self, node_type_id: str) -> bool:
        """Remove a node definition if present."""
        return self.definitions.pop(node_type_id, None) is not None

    def get_definition(self, node_type_id: str) -> Optional[NodeDefinition]:
        return self.definitions.get(node_type_id)

    def register_functor_instance(self, functor: Any, node_type_id: Optional[str] = None) -> NodeDefinition:
        metadata = getattr(functor, "metadata", None)
        if metadata is None:
            raise ValueError("Functor instance has no metadata.")
        input_names = _normalize_names(getattr(metadata, "input_variables", None))
        output_names = _normalize_names(getattr(metadata, "output_variables", None))
        get_required_inputs = getattr(functor, "get_required_inputs", None)
        if not callable(get_required_inputs):
            raise TypeError(
                f"Functor '{functor.__class__.__name__}' has no get_required_inputs(); "
                "required inputs are read only from the FunctorBase rule."
            )
        required_input_names = _normalize_names(get_required_inputs())
        undeclared = [
            name for name in required_input_names if name not in input_names
        ]
        if undeclared:
            raise ValueError(
                f"Functor '{functor.__class__.__name__}' required input "
                f"{undeclared!r} is not declared in metadata.input_variables."
            )
        node_type = node_type_id or f"Functor:{functor.__class__.__module__}.{functor.__class__.__name__}"
        definition = self._build_definition(
            node_type,
            input_names,
            output_names,
            metadata_dict=getattr(metadata, "to_dict", None),
            required_input_names=required_input_names,
        )
        definition.metadata["runtime_class"] = (
            f"{functor.__class__.__module__}.{functor.__class__.__name__}"
        )
        self.register_definition(node_type, definition)
        return definition

    def remove_definitions_by_functor_names(self, functor_names: Iterable[str]) -> List[str]:
        """
        Remove node definitions whose metadata name matches provided functor names.
        """
        targets = {str(name) for name in functor_names}
        removed: List[str] = []
        for node_type_id, definition in list(self.definitions.items()):
            metadata_name = str(definition.metadata.get("name", ""))
            if metadata_name in targets:
                if self.unregister_definition(node_type_id):
                    removed.append(node_type_id)
        return removed

    def build_from_node_spec(
        self,
        node_type_id: str,
        input_names: Iterable[str],
        output_names: Iterable[str],
    ) -> NodeDefinition:
        return self._build_definition(node_type_id, list(input_names), list(output_names))

    def _build_definition(
        self,
        node_type_id: str,
        input_names: List[str],
        output_names: List[str],
        metadata_dict: Any = None,
        required_input_names: Optional[Iterable[str]] = None,
    ) -> NodeDefinition:
        inputs = self._build_ports(
            input_names,
            required=True,
            required_names=required_input_names,
            direction="in",
        )
        outputs = self._build_ports(
            output_names,
            required=False,
            direction="out",
        )
        metadata_payload: Dict[str, Any] = {}
        if callable(metadata_dict):
            captured = metadata_dict()
            if not isinstance(captured, Mapping):
                raise TypeError("Functor metadata must serialize to a mapping.")
            metadata_payload = deepcopy(dict(captured))
        elif metadata_dict is not None:
            raise TypeError("Functor metadata serializer must be callable.")
        return NodeDefinition(node_type_id=node_type_id, inputs=inputs, outputs=outputs, metadata=metadata_payload)

    def _build_ports(
        self,
        names: List[str],
        required: bool,
        required_names: Optional[Iterable[str]] = None,
        direction: str = "inout",
    ) -> Dict[str, PortSpec]:
        ports: Dict[str, PortSpec] = {}
        required_name_set = (
            None if required_names is None else set(required_names)
        )
        for name in names:
            port_required = (
                required
                if required_name_set is None
                else name in required_name_set
            )
            port = self._build_port(
                name,
                required=port_required,
                direction=direction,
            )
            ports[name] = port
        return ports

    def _build_port(
        self,
        name: str,
        required: bool,
        direction: str,
    ) -> PortSpec:
        standard_name = variable_name_mapper.to_standard(name)
        entry = self._resolve_entry(name, standard_name)
        if entry is None:
            standard_name = standard_name or name
            units = None
            quantity = None
            cf_standard_name = None
            element = None
            semantics = None
            update_mode = None
            write_mode = None
            context_requirements: Dict[str, str] = {}
            datatype = None
            shape = None
            port_metadata: Dict[str, Any] = {}
        else:
            standard_name = entry.standard_name
            units = entry.units
            quantity = entry.quantity
            cf_standard_name = SSOTCatalog._normalize_optional_cf_standard_name(
                entry.cf_standard_name,
                source=f"SSOT entry '{entry.name}'",
            )
            element = entry.element
            semantics = entry.semantics
            update_mode = entry.update_mode
            write_mode = entry.write_mode
            context_requirements = dict(entry.context_requirements)
            datatype = entry.datatype
            shape = entry.shape
            port_metadata = self._extract_port_metadata(entry)

        payload: Dict[str, Any] = {
            "name": name,
            "standard_name": standard_name,
            "units": units,
            "quantity": quantity,
            "cf_standard_name": cf_standard_name,
            "element": element,
            "semantics": semantics,
            "update_mode": update_mode,
            "write_mode": None if direction == "in" else write_mode,
            "required": required,
            "direction": direction,
            "context_requirements": context_requirements,
            "shape": shape,
            **port_metadata,
        }
        if datatype:
            payload["datatype"] = datatype
        return PortSpec(**payload)

    @staticmethod
    def _extract_port_metadata(entry: SSOTEntry) -> Dict[str, Any]:
        raw = entry.raw if isinstance(entry.raw, dict) else {}
        form_keys = (
            "form_degree",
            "primality",
            "extensivity",
            "complex_id",
            "physical_domain",
            "orientation_convention",
            "boundary_role",
            "port_kind",
            "power_conjugate",
        )
        metadata = {
            key: raw.get(key)
            for key in form_keys
            if raw.get(key) is not None
        }
        support = raw.get("support")
        if support is not None and not isinstance(support, dict):
            raise ValueError(
                f"SSOT entry '{entry.name}' support metadata must be a mapping."
            )
        if isinstance(support, dict):
            declared_support_fields = {
                key for key in PORT_SUPPORT_FIELDS if key in support
            }
            if "shape" in raw:
                declared_support_fields.add("shape")
            for key in PORT_SUPPORT_FIELDS[1:]:
                if key in support:
                    metadata[key] = support[key]
            metadata["declared_support_fields"] = frozenset(
                declared_support_fields
            )
        return metadata

    def _resolve_entry(self, name: str, standard_name: str) -> Optional[SSOTEntry]:
        entry = self.ssot_catalog.get(name)
        if entry is not None:
            return entry
        if standard_name and standard_name != name:
            return self.ssot_catalog.get_by_standard_name(standard_name)
        return None

    def resolve_class_type(self, class_type: str) -> Optional[type]:
        definition = self.get_definition(class_type)
        if definition is not None:
            runtime_class = definition.metadata.get("runtime_class")
            if runtime_class is not None:
                if not isinstance(runtime_class, str) or "." not in runtime_class:
                    raise ValueError(
                        f"Node definition '{class_type}' has invalid runtime_class."
                    )
                module_path, class_name = runtime_class.rsplit(".", 1)
                module = import_module(module_path)
                resolved = getattr(module, class_name, None)
                if not isinstance(resolved, type):
                    raise TypeError(
                        f"Node definition '{class_type}' runtime_class "
                        f"'{runtime_class}' is not a class."
                    )
                return resolved
        if not class_type.startswith("Functor:"):
            return None
        dotted = class_type.split(":", 1)[-1].strip()
        if not dotted or "." not in dotted:
            return None
        module_path, class_name = dotted.rsplit(".", 1)
        module = import_module(module_path)
        resolved = getattr(module, class_name, None)
        if isinstance(resolved, type):
            return resolved
        return None
