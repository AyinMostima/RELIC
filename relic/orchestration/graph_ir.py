"""
Graph IR structures for orchestration.

Provides minimal dataclasses for Graph IR parsing and serialization.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LinkEndpoint:
    node: str
    port: str


@dataclass
class LinkSpec:
    source: LinkEndpoint
    target: LinkEndpoint


@dataclass
class NodeSpec:
    class_type: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphIR:
    schema_version: str
    nodes: Dict[str, NodeSpec]
    links: List[LinkSpec] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    assertion: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphIR":
        if not isinstance(payload, dict):
            raise TypeError("Graph IR payload must be a dictionary.")
        schema_version = str(payload.get("schema_version", "")).strip()
        if not schema_version:
            raise ValueError("Graph IR requires schema_version.")

        raw_nodes = payload.get("nodes", {})
        if not isinstance(raw_nodes, dict):
            raise TypeError("Graph IR nodes must be a mapping.")

        nodes: Dict[str, NodeSpec] = {}
        for node_id, node_payload in raw_nodes.items():
            if not isinstance(node_payload, dict):
                raise TypeError(f"Node '{node_id}' must be a mapping.")
            class_type = str(node_payload.get("class_type", "")).strip()
            if not class_type:
                raise ValueError(f"Node '{node_id}' missing class_type.")
            inputs = node_payload.get("inputs", {}) or {}
            outputs = node_payload.get("outputs", []) or []
            config = node_payload.get("config", {}) or {}
            if not isinstance(inputs, dict):
                raise TypeError(f"Node '{node_id}' inputs must be a mapping.")
            if not isinstance(outputs, list):
                raise TypeError(f"Node '{node_id}' outputs must be a list.")
            nodes[str(node_id)] = NodeSpec(
                class_type=class_type,
                inputs=dict(inputs),
                outputs=[str(item) for item in outputs],
                config=dict(config),
            )

        links = cls._parse_links(payload.get("links"), nodes)
        metadata = payload.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            raise TypeError("Graph IR metadata must be a mapping.")
        assertion = payload.get("assertion", {}) or {}
        if not isinstance(assertion, dict):
            raise TypeError("Graph IR assertion must be a mapping.")

        graph = cls(
            schema_version=schema_version,
            nodes=nodes,
            links=links,
            metadata=dict(metadata),
            assertion=dict(assertion),
        )
        if not graph.links:
            graph.links = graph._infer_links_from_inputs()
        return graph

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "nodes": {
                node_id: {
                    "class_type": node.class_type,
                    "inputs": dict(node.inputs),
                    "outputs": list(node.outputs),
                    "config": dict(node.config),
                }
                for node_id, node in self.nodes.items()
            },
            "links": [
                {
                    "from": {"node": link.source.node, "port": link.source.port},
                    "to": {"node": link.target.node, "port": link.target.port},
                }
                for link in self.links
            ],
            "metadata": dict(self.metadata),
        }
        if self.assertion:
            payload["assertion"] = dict(self.assertion)
        return payload

    def required_initial_state_keys(self) -> List[str]:
        """Return state keys that the declarative graph requires at startup."""

        required_keys: set[str] = set()
        external_owners: Dict[str, str] = {}
        for node_id, node in self.nodes.items():
            if node.class_type == "State:ExternalForcing":
                if not node.outputs:
                    raise ValueError(
                        f"External forcing boundary '{node_id}' must declare "
                        "at least one output."
                    )
                seen_outputs: set[str] = set()
                for raw_output in node.outputs:
                    if not isinstance(raw_output, str) or not raw_output.strip():
                        raise ValueError(
                            f"External forcing boundary '{node_id}' must declare "
                            "non-empty outputs."
                        )
                    if raw_output != raw_output.strip():
                        raise ValueError(
                            f"External forcing output '{node_id}.{raw_output}' must "
                            "not contain leading or trailing whitespace."
                        )
                    output = raw_output
                    if output in seen_outputs:
                        raise ValueError(
                            f"External forcing boundary '{node_id}' has duplicate "
                            f"output '{output}'."
                        )
                    seen_outputs.add(output)
                    existing_owner = external_owners.get(output)
                    if existing_owner is not None:
                        raise ValueError(
                            f"External state key '{output}' is declared by both "
                            f"'{existing_owner}' and '{node_id}'."
                        )
                    external_owners[output] = node_id
                    required_keys.add(output)

            for input_name, binding in node.inputs.items():
                if not isinstance(binding, Mapping) or "state" not in binding:
                    continue
                state_key = binding.get("state")
                if not isinstance(state_key, str) or not state_key.strip():
                    raise ValueError(
                        f"State binding '{node_id}.{input_name}' must name a "
                        "non-empty state key."
                    )
                if state_key != state_key.strip():
                    raise ValueError(
                        f"State binding '{node_id}.{input_name}' must not contain "
                        "leading or trailing whitespace."
                    )
                required_keys.add(state_key)
        return sorted(required_keys)

    @staticmethod
    def _parse_links(raw_links: Any, nodes: Dict[str, NodeSpec]) -> List[LinkSpec]:
        if raw_links is None:
            return []
        if not isinstance(raw_links, list):
            raise TypeError("Graph IR links must be a list.")
        links: List[LinkSpec] = []
        for idx, link_payload in enumerate(raw_links):
            if not isinstance(link_payload, dict):
                raise TypeError(f"links[{idx}] must be a mapping.")
            source_payload = link_payload.get("from", {})
            target_payload = link_payload.get("to", {})
            if not isinstance(source_payload, dict) or not isinstance(target_payload, dict):
                raise TypeError(f"links[{idx}] must contain mapping for from/to.")
            source = LinkEndpoint(
                node=str(source_payload.get("node", "")),
                port=str(source_payload.get("port", "")),
            )
            target = LinkEndpoint(
                node=str(target_payload.get("node", "")),
                port=str(target_payload.get("port", "")),
            )
            links.append(LinkSpec(source=source, target=target))
        return links

    def _infer_links_from_inputs(self) -> List[LinkSpec]:
        inferred: List[LinkSpec] = []
        for node_id, node in self.nodes.items():
            for port_name, payload in node.inputs.items():
                if not isinstance(payload, dict):
                    continue
                link_payload = payload.get("link")
                if not isinstance(link_payload, (list, tuple)) or len(link_payload) != 2:
                    continue
                source_node, source_port = link_payload
                inferred.append(
                    LinkSpec(
                        source=LinkEndpoint(node=str(source_node), port=str(source_port)),
                        target=LinkEndpoint(node=str(node_id), port=str(port_name)),
                    )
                )
        return inferred
