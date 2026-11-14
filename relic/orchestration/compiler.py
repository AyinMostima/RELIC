"""
Graph compiler for orchestration.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields as dataclass_fields, is_dataclass
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

import torch

from ..data.standards.unit_converter import ConversionContext
from .adapters import AdapterRegistry, ExecutableAdapterFunctor
from .graph_ir import GraphIR, LinkEndpoint, LinkSpec, NodeSpec
from .node_registry import NodeRegistry
from .port_spec import PortSpec
from .validator import (
    ValidationIssue,
    ValidationReport,
    _required_iphs_level,
    build_audit_summary,
    can_connect,
    validate_graph,
)


@dataclass
class AdapterInsertion:
    adapter_type: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionStep:
    node_id: str
    class_type: str
    inputs: dict[str, Any]
    outputs: list[str]
    config: dict[str, Any]


@dataclass(frozen=True)
class _IPHSCompileIdentityBinding:
    """Compile-local identity binding that GraphIR payloads cannot mint."""

    source_node_id: str
    node_type_id: str
    runtime_class: str
    runtime_owner: str
    node_definition_sha256: str
    execution_step_sha256: str
    port_contract_status: str

    def to_audit_dict(self) -> dict[str, str]:
        return {
            "source_node_id": self.source_node_id,
            "node_type_id": self.node_type_id,
            "runtime_class": self.runtime_class,
            "runtime_owner": self.runtime_owner,
            "binding_kind": "compile_local_execution_step_identity",
            "node_definition_sha256": self.node_definition_sha256,
            "execution_step_sha256": self.execution_step_sha256,
            "port_contract_status": self.port_contract_status,
        }


@dataclass
class CompileReport:
    validation: ValidationReport
    adapter_insertions: list[AdapterInsertion] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


_PLAN_SEAL_SCHEMA_VERSION = "1.0.0"
_PLAN_SEAL_ISSUER = "relic.orchestration.compile_graph"


@dataclass(frozen=True)
class _ExecutionPlanSeal:
    """Content commitment issued only by the canonical graph compiler."""

    schema_version: str
    issuer: str
    phase: str
    source_graph_sha256: str
    transformed_graph_sha256: str
    executable_graph_sha256: str
    source_registry_contract_sha256: str
    registry_contract_sha256: str
    compiler_options_sha256: str
    plan_sha256: str
    overall_sha256: str
    source_node_types: tuple[str, ...]
    referenced_node_types: tuple[str, ...]
    registry_bound_node_types: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "issuer": self.issuer,
            "phase": self.phase,
            "source_graph_sha256": self.source_graph_sha256,
            "transformed_graph_sha256": self.transformed_graph_sha256,
            "executable_graph_sha256": self.executable_graph_sha256,
            "source_registry_contract_sha256": (
                self.source_registry_contract_sha256
            ),
            "registry_contract_sha256": self.registry_contract_sha256,
            "compiler_options_sha256": self.compiler_options_sha256,
            "plan_sha256": self.plan_sha256,
            "overall_sha256": self.overall_sha256,
            "source_node_types": list(self.source_node_types),
            "referenced_node_types": list(self.referenced_node_types),
            "registry_bound_node_types": list(
                self.registry_bound_node_types
            ),
        }


@dataclass
class ExecutionPlan:
    steps: list[ExecutionStep]
    report: CompileReport
    _compiler_seal: _ExecutionPlanSeal | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


def _type_identity(value: type[Any] | Any) -> dict[str, str]:
    value_type = value if isinstance(value, type) else type(value)
    return {
        "module": value_type.__module__,
        "qualname": value_type.__qualname__,
    }


def _canonicalize_plan_value(
    value: Any,
    *,
    path: str = "$",
    active_ids: set[int] | None = None,
) -> Any:
    """Convert compiler inputs to an unambiguous JSON-safe value tree."""

    active = active_ids if active_ids is not None else set()
    if isinstance(value, Enum):
        return {
            "type": "enum",
            "class": _type_identity(value),
            "name": value.name,
            "value": _canonicalize_plan_value(
                value.value,
                path=f"{path}.value",
                active_ids=active,
            ),
        }
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        encoded = value.hex()
        if encoded in {"nan", "inf", "-inf"}:
            raise ValueError(f"Non-finite float is not sealable at {path}.")
        return {"type": "float", "hex": encoded}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, Path):
        return {"type": "path", "value": value.as_posix()}
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    if isinstance(value, bytearray):
        return {"type": "bytearray", "hex": bytes(value).hex()}
    if isinstance(value, torch.dtype):
        return {"type": "torch.dtype", "value": str(value)}
    if isinstance(value, torch.device):
        return {"type": "torch.device", "value": str(value)}
    if isinstance(value, torch.layout):
        return {"type": "torch.layout", "value": str(value)}
    if isinstance(value, torch.Tensor):
        if value.layout is not torch.strided:
            raise ValueError(
                f"Only dense strided tensors are sealable at {path}."
            )
        if value.is_quantized:
            raise ValueError(f"Quantized tensor is not sealable at {path}.")
        if value.device.type == "meta":
            raise ValueError(f"Meta tensor is not sealable at {path}.")
        if not value.is_leaf or value.grad_fn is not None:
            raise ValueError(f"A non-leaf tensor is not sealable at {path}.")
        if torch.is_floating_point(value) or torch.is_complex(value):
            finite = torch.all(torch.isfinite(value))
            if not bool(finite.detach().to(device="cpu").item()):
                raise ValueError(
                    f"Non-finite tensor is not sealable at {path}."
                )
        try:
            materialized = (
                value.detach()
                .resolve_conj()
                .resolve_neg()
                .to(device="cpu")
                .contiguous()
                .reshape(-1)
            )
            raw_bytes = bytes(materialized.view(torch.uint8).tolist()).hex()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Tensor dtype is not sealable at {path}: {value.dtype}."
            ) from exc
        return {
            "type": "torch.Tensor",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "device": str(value.device),
            "layout": str(value.layout),
            "requires_grad": value.requires_grad,
            "byteorder": sys.byteorder,
            "data_hex": raw_bytes,
        }
    if isinstance(value, type):
        return {"type": "python.type", "identity": _type_identity(value)}

    tracked = isinstance(
        value,
        (Mapping, list, tuple, set, frozenset),
    ) or (is_dataclass(value) and not isinstance(value, type))
    identity = id(value)
    if tracked:
        if identity in active:
            raise ValueError(f"Cyclic value is not sealable at {path}.")
        active.add(identity)
    try:
        if is_dataclass(value) and not isinstance(value, type):
            return {
                "type": "dataclass",
                "class": _type_identity(value),
                "fields": {
                    item.name: _canonicalize_plan_value(
                        getattr(value, item.name),
                        path=f"{path}.{item.name}",
                        active_ids=active,
                    )
                    for item in dataclass_fields(value)
                },
            }
        if isinstance(value, Mapping):
            entries: list[dict[str, Any]] = []
            for key, item_value in value.items():
                canonical_key = _canonicalize_plan_value(
                    key,
                    path=f"{path}.<key>",
                    active_ids=active,
                )
                canonical_value = _canonicalize_plan_value(
                    item_value,
                    path=f"{path}[{key!s}]",
                    active_ids=active,
                )
                entries.append(
                    {"key": canonical_key, "value": canonical_value}
                )
            entries.sort(
                key=lambda item: json.dumps(
                    item["key"],
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            return {"type": "mapping", "entries": entries}
        if isinstance(value, (list, tuple)):
            return {
                "type": "tuple" if isinstance(value, tuple) else "list",
                "items": [
                    _canonicalize_plan_value(
                        item,
                        path=f"{path}[{index}]",
                        active_ids=active,
                    )
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, (set, frozenset)):
            items = [
                _canonicalize_plan_value(
                    item,
                    path=f"{path}.<set-item>",
                    active_ids=active,
                )
                for item in value
            ]
            items.sort(
                key=lambda item: json.dumps(
                    item,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            return {
                "type": (
                    "frozenset" if isinstance(value, frozenset) else "set"
                ),
                "items": items,
            }
    finally:
        if tracked:
            active.remove(identity)
    raise TypeError(
        f"Unsupported compiler seal value at {path}: "
        f"{type(value).__module__}.{type(value).__qualname__}."
    )


def _canonical_json_bytes(value: Any) -> bytes:
    canonical = _canonicalize_plan_value(value)
    return json.dumps(
        canonical,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _node_types_from_graph(graph: GraphIR) -> tuple[str, ...]:
    return tuple(sorted({node.class_type for node in graph.nodes.values()}))


def _registry_contract_payload(
    registry: NodeRegistry | None,
    node_types: tuple[str, ...],
) -> dict[str, Any]:
    if tuple(sorted(set(node_types))) != node_types:
        raise ValueError("Registry contract node types must be sorted and unique.")
    if node_types and registry is None:
        raise ValueError(
            "Registry contract verification requires the canonical NodeRegistry."
        )
    entries: list[dict[str, Any]] = []
    for node_type in node_types:
        if not isinstance(node_type, str) or not node_type:
            raise ValueError("Registry contract node types must be non-empty strings.")
        definition = (
            None if registry is None else registry.get_definition(node_type)
        )
        if definition is None:
            entries.append({"lookup_key": node_type, "present": False})
            continue
        if definition.node_type_id != node_type:
            raise ValueError(
                "Registry contract lookup key does not match "
                f"NodeDefinition.node_type_id for '{node_type}'."
            )
        entries.append(
            {
                "lookup_key": node_type,
                "present": True,
                "definition": definition.to_dict(),
            }
        )
    return {
        "schema_version": "1.0.0",
        "node_types": entries,
    }


def _registry_bound_node_types(
    registry: NodeRegistry | None,
    node_types: tuple[str, ...],
) -> tuple[str, ...]:
    if registry is None:
        return ()
    return tuple(
        node_type
        for node_type in node_types
        if registry.get_definition(node_type) is not None
    )


def _optimization_pass_contract(opt_pass: OptimizationPass) -> dict[str, Any]:
    state = dict(vars(opt_pass)) if hasattr(opt_pass, "__dict__") else {}
    return {
        "implementation": _type_identity(opt_pass),
        "name": getattr(opt_pass, "name", None),
        "state": state,
    }


def _adapter_registry_contract(adapter_registry: AdapterRegistry) -> dict[str, Any]:
    adapters = getattr(adapter_registry, "_adapters", {})
    if not isinstance(adapters, Mapping):
        raise TypeError("AdapterRegistry adapter contract must be a mapping.")
    return {
        "implementation": _type_identity(adapter_registry),
        "unit_converter": _type_identity(adapter_registry.unit_converter),
        "adapters": [
            {
                "adapter_type": str(adapter_type),
                "implementation": _type_identity(adapter),
            }
            for adapter_type, adapter in sorted(
                adapters.items(),
                key=lambda item: str(item[0]),
            )
        ],
    }


def _execution_plan_content_payload(plan: ExecutionPlan) -> dict[str, Any]:
    return {
        "steps": plan.steps,
        "report": plan.report,
    }


def _seal_payload_without_overall(seal: _ExecutionPlanSeal) -> dict[str, Any]:
    payload = seal.to_dict()
    payload.pop("overall_sha256")
    return payload


def _mint_execution_plan_seal(
    plan: ExecutionPlan,
    *,
    source_graph_sha256: str,
    transformed_graph_sha256: str,
    executable_graph_sha256: str,
    source_registry_contract_sha256: str,
    registry_contract_sha256: str,
    compiler_options_sha256: str,
    source_node_types: tuple[str, ...],
    referenced_node_types: tuple[str, ...],
    registry_bound_node_types: tuple[str, ...],
) -> _ExecutionPlanSeal:
    unsigned = _ExecutionPlanSeal(
        schema_version=_PLAN_SEAL_SCHEMA_VERSION,
        issuer=_PLAN_SEAL_ISSUER,
        phase="final",
        source_graph_sha256=source_graph_sha256,
        transformed_graph_sha256=transformed_graph_sha256,
        executable_graph_sha256=executable_graph_sha256,
        source_registry_contract_sha256=(
            source_registry_contract_sha256
        ),
        registry_contract_sha256=registry_contract_sha256,
        compiler_options_sha256=compiler_options_sha256,
        plan_sha256=_canonical_sha256(
            _execution_plan_content_payload(plan)
        ),
        overall_sha256="",
        source_node_types=source_node_types,
        referenced_node_types=referenced_node_types,
        registry_bound_node_types=registry_bound_node_types,
    )
    return _ExecutionPlanSeal(
        **{
            **_seal_payload_without_overall(unsigned),
            "source_node_types": unsigned.source_node_types,
            "referenced_node_types": unsigned.referenced_node_types,
            "registry_bound_node_types": (
                unsigned.registry_bound_node_types
            ),
            "overall_sha256": _canonical_sha256(
                _seal_payload_without_overall(unsigned)
            ),
        }
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_execution_plan_content(
    plan: ExecutionPlan,
) -> _ExecutionPlanSeal:
    """Validate the registry-independent portion of the compiler seal."""
    if type(plan) is not ExecutionPlan:
        raise TypeError("Execution plan must use the canonical ExecutionPlan type.")
    seal = plan._compiler_seal
    if type(seal) is not _ExecutionPlanSeal:
        raise ValueError("Execution plan has no canonical compiler seal.")
    if seal.schema_version != _PLAN_SEAL_SCHEMA_VERSION:
        raise ValueError("Execution plan compiler seal schema is unsupported.")
    if seal.issuer != _PLAN_SEAL_ISSUER:
        raise ValueError("Execution plan compiler seal issuer is invalid.")
    if seal.phase != "final":
        raise ValueError("Execution plan compiler seal is not final.")
    digest_fields = (
        seal.source_graph_sha256,
        seal.transformed_graph_sha256,
        seal.executable_graph_sha256,
        seal.source_registry_contract_sha256,
        seal.registry_contract_sha256,
        seal.compiler_options_sha256,
        seal.plan_sha256,
        seal.overall_sha256,
    )
    if not all(_is_sha256(value) for value in digest_fields):
        raise ValueError("Execution plan compiler seal contains an invalid digest.")
    for label, node_types in (
        ("source", seal.source_node_types),
        ("referenced", seal.referenced_node_types),
        ("registry-bound", seal.registry_bound_node_types),
    ):
        if type(node_types) is not tuple or tuple(sorted(set(node_types))) != node_types or any(
            not isinstance(node_type, str) or not node_type
            for node_type in node_types
        ):
            raise ValueError(
                f"Execution plan {label} node types are not canonical."
            )
    if type(plan.steps) is not list or type(plan.report) is not CompileReport:
        raise ValueError("Execution plan content is not canonical.")
    if any(type(step) is not ExecutionStep for step in plan.steps):
        raise ValueError("Execution plan content contains a non-canonical step.")
    step_node_types = {step.class_type for step in plan.steps}
    if not step_node_types.issubset(set(seal.referenced_node_types)):
        raise ValueError(
            "Execution plan referenced node types do not cover plan content."
        )
    if not set(seal.source_node_types).issubset(
        set(seal.referenced_node_types)
    ):
        raise ValueError(
            "Execution plan referenced node types do not cover the source graph."
        )
    actual_plan_sha256 = _canonical_sha256(
        _execution_plan_content_payload(plan)
    )
    if actual_plan_sha256 != seal.plan_sha256:
        raise ValueError("Execution plan content does not match its compiler seal.")
    if _canonical_sha256(_seal_payload_without_overall(seal)) != seal.overall_sha256:
        raise ValueError("Execution plan compiler seal overall digest is invalid.")
    return seal


def verify_execution_plan_integrity(
    plan: ExecutionPlan,
    registry: NodeRegistry | None,
) -> dict[str, Any]:
    """Fail closed unless a live plan matches its compiler-issued seal."""

    seal = _verify_execution_plan_content(plan)
    source_registry_sha256 = _canonical_sha256(
        _registry_contract_payload(registry, seal.source_node_types)
    )
    if source_registry_sha256 != seal.source_registry_contract_sha256:
        raise ValueError(
            "Execution plan source registry contract has changed."
        )
    registry_sha256 = _canonical_sha256(
        _registry_contract_payload(registry, seal.referenced_node_types)
    )
    if registry_sha256 != seal.registry_contract_sha256:
        raise ValueError("Execution plan registry contract has changed.")
    actual_bound_types = _registry_bound_node_types(
        registry,
        seal.referenced_node_types,
    )
    if actual_bound_types != seal.registry_bound_node_types:
        raise ValueError(
            "Execution plan registry contract bound types have changed."
        )
    return seal.to_dict()


@dataclass
class BoundedSynthesisResult:
    """Deterministic finite synthesis result owned by the canonical compiler."""

    request_id: str
    enumerated_count: int
    valid_graphs: dict[str, GraphIR]
    invalid_reasons: dict[str, str]
    invalid_issues: dict[str, list[dict[str, Any]]]

    @property
    def valid_count(self) -> int:
        return len(self.valid_graphs)

    @property
    def invalid_count(self) -> int:
        return len(self.invalid_reasons)

    @property
    def complete_enumeration(self) -> bool:
        return self.enumerated_count == self.valid_count + self.invalid_count

    def to_dict(self) -> dict[str, Any]:
        graph_hashes = {
            selection_id: hashlib.sha256(
                json.dumps(
                    graph.to_dict(),
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for selection_id, graph in sorted(self.valid_graphs.items())
        }
        return {
            "schema_version": "1.0.0",
            "document_kind": "bounded_graph_synthesis_result",
            "request_id": self.request_id,
            "enumerated_count": self.enumerated_count,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "complete_enumeration": self.complete_enumeration,
            "valid_selection_ids": sorted(self.valid_graphs),
            "invalid_reasons": dict(sorted(self.invalid_reasons.items())),
            "invalid_issues": {
                selection_id: [
                    dict(issue)
                    for issue in issues
                ]
                for selection_id, issues in sorted(
                    self.invalid_issues.items()
                )
            },
            "valid_graph_sha256": graph_hashes,
            "precision_against_oracle": None,
            "recall_against_oracle": None,
        }


@runtime_checkable
class OptimizationPass(Protocol):
    """
    Protocol for graph optimization passes.

    References
    ----------
    NVIDIA OmniGraph style optimization transforms.

    Mathematical Expression
    -----------------------
    G' = pass(G) where |G'| <= |G| and semantics(G') = semantics(G).
    """

    name: str

    def apply(self, graph: GraphIR, registry: NodeRegistry) -> GraphIR:
        """Apply optimization pass to the graph."""
        ...


class IdentityEliminationPass:
    """
    Eliminate identity or passthrough nodes that do not transform data.

    References
    ----------
    Common compiler optimization: identity elimination.

    Mathematical Expression
    -----------------------
    Remove node n if output(n) = input(n) without transformation.
    """

    name = "identity_elimination"

    def apply(self, graph: GraphIR, registry: NodeRegistry) -> GraphIR:
        _ = registry
        identity_tokens = {"identity", "passthrough"}
        nodes_to_remove: set[str] = set()
        redirect_map: dict[str, tuple[str, str]] = {}

        for node_id, node in graph.nodes.items():
            class_type = node.class_type.lower()
            is_identity = any(token in class_type for token in identity_tokens)
            if not is_identity or len(node.inputs) != 1:
                continue
            input_spec = next(iter(node.inputs.values()))
            if not isinstance(input_spec, dict) or "link" not in input_spec:
                continue
            link = input_spec.get("link")
            if isinstance(link, (list, tuple)) and len(link) == 2:
                source_node, source_port = link
                nodes_to_remove.add(node_id)
                redirect_map[node_id] = (
                    str(source_node),
                    str(source_port),
                )

        if not nodes_to_remove:
            return graph

        new_nodes: dict[str, NodeSpec] = {}
        for node_id, node in graph.nodes.items():
            if node_id in nodes_to_remove:
                continue
            new_inputs: dict[str, Any] = {}
            for input_name, input_spec in node.inputs.items():
                if isinstance(input_spec, dict) and "link" in input_spec:
                    link = input_spec.get("link")
                    if isinstance(link, (list, tuple)) and len(link) == 2:
                        source_node, source_port = link
                        while source_node in redirect_map:
                            source_node, source_port = redirect_map[source_node]
                        new_inputs[input_name] = {"link": [source_node, source_port]}
                        continue
                new_inputs[input_name] = input_spec
            new_nodes[node_id] = NodeSpec(
                class_type=node.class_type,
                inputs=new_inputs,
                outputs=list(node.outputs),
                config=dict(node.config),
            )

        links = graph.links or graph._infer_links_from_inputs()
        new_links: list[LinkSpec] = []
        for link in links:
            source = link.source.node
            source_port = link.source.port
            target = link.target.node
            while source in redirect_map:
                source, source_port = redirect_map[source]
            if target in nodes_to_remove:
                continue
            new_links.append(
                LinkSpec(
                    source=LinkEndpoint(node=source, port=source_port),
                    target=link.target,
                )
            )

        return GraphIR(
            schema_version=graph.schema_version,
            nodes=new_nodes,
            links=new_links,
            metadata={
                **graph.metadata,
                "optimizations_applied": graph.metadata.get("optimizations_applied", [])
                + [self.name],
            },
            assertion=dict(graph.assertion),
        )


class ConstantFoldingPass:
    """
    Fold constant nodes into their consumers.

    References
    ----------
    Common compiler optimization: constant folding.

    Mathematical Expression
    -----------------------
    If node n produces constant c, replace references to n with c.
    """

    name = "constant_folding"

    def apply(self, graph: GraphIR, registry: NodeRegistry) -> GraphIR:
        _ = registry
        constant_tokens = {"constant", "literal"}
        constant_nodes: dict[str, Any] = {}

        for node_id, node in graph.nodes.items():
            class_type = node.class_type.lower()
            if not any(token in class_type for token in constant_tokens):
                continue
            if "value" in node.config:
                constant_nodes[node_id] = node.config.get("value")

        if not constant_nodes:
            return graph

        new_nodes: dict[str, NodeSpec] = {}
        for node_id, node in graph.nodes.items():
            if node_id in constant_nodes:
                continue
            new_inputs: dict[str, Any] = {}
            for input_name, input_spec in node.inputs.items():
                if isinstance(input_spec, dict) and "link" in input_spec:
                    link = input_spec.get("link")
                    if isinstance(link, (list, tuple)) and len(link) == 2:
                        source_node = str(link[0])
                        if source_node in constant_nodes:
                            new_inputs[input_name] = {"value": constant_nodes[source_node]}
                            continue
                new_inputs[input_name] = input_spec
            new_nodes[node_id] = NodeSpec(
                class_type=node.class_type,
                inputs=new_inputs,
                outputs=list(node.outputs),
                config=dict(node.config),
            )

        links = graph.links or graph._infer_links_from_inputs()
        new_links = [
            link for link in links if link.source.node not in constant_nodes
        ]

        return GraphIR(
            schema_version=graph.schema_version,
            nodes=new_nodes,
            links=new_links,
            metadata={
                **graph.metadata,
                "optimizations_applied": graph.metadata.get("optimizations_applied", [])
                + [self.name],
            },
            assertion=dict(graph.assertion),
        )


DEFAULT_OPTIMIZATION_PASSES: list[OptimizationPass] = [
    IdentityEliminationPass(),
    ConstantFoldingPass(),
]

_STRUCTURAL_AUDIT_KEYS = (
    "dec_topology_audit",
    "hodge_audit",
    "iphs_audit",
    "entropy_audit",
    "port_power_audit",
    "port_matching_audit",
    "overflow_routing_audit",
    "equivalence_certificate_audit",
    "reaction_network_audit",
    "stochastic_sdae_audit",
    "integration_plan_audit",
    "state_domain_audit",
    "observation_audit",
    "naturality_audit",
    "reproducibility_audit",
    "scientific_migration_audit",
)


def _empty_structural_audit() -> dict[str, Any]:
    return {"enabled": False, "passed": None, "source": "not_declared"}


def _normalize_audit_payload(payload: Any, *, source: str) -> dict[str, Any]:
    if isinstance(payload, dict):
        normalized = dict(payload)
        normalized.setdefault("enabled", True)
        normalized.setdefault("source", source)
        return normalized
    if isinstance(payload, bool):
        return {"enabled": True, "passed": payload, "source": source}
    return {"enabled": bool(payload), "passed": None, "source": source}


def _normalize_unverified_audit_declaration(payload: Any, *, source: str) -> dict[str, Any]:
    declared = _normalize_audit_payload(payload, source=source)
    declared_passed = declared.get("passed")
    declared.update(
        {
            "enabled": False,
            "passed": None,
            "verified": False,
            "declared_passed": declared_passed,
            "source": f"{source}.unverified_declaration",
        }
    )
    return declared


def _resolve_iphs_assembly_payload(
    payload: Mapping[str, Any],
    graph: GraphIR | None,
) -> dict[str, Any]:
    """Resolve explicit GraphIR IPHS assembly declarations.

    References:
      - van der Schaft and Maschke (1995), port-Hamiltonian interconnection.
      - Grmela and Oettinger (1997), GENERIC reversible/irreversible split.
    Mathematical expression:
      Global assembly consumes declared local contributions i_i J_i i_i^T,
      i_i M_i i_i^T, i_i G_i, and i_i N_i.
    Parameter meanings:
      payload: graph.metadata.iphs_assembly mapping.
      graph: existing GraphIR carrying node config declarations.
    """

    resolved = dict(payload)
    contributions: list[Any] = list(
        resolved.get("contributions", ()) or ()
    )
    contribution_origins: list[str] = [
        "inline_metadata" for _ in contributions
    ]
    node_ids_payload = resolved.get("contribution_node_ids")
    collect_all_nodes = bool(resolved.get("collect_node_contributions", False))
    collected_count = 0

    if graph is not None and (collect_all_nodes or node_ids_payload is not None):
        if node_ids_payload is None:
            node_ids = sorted(graph.nodes.keys())
            require_each_declared = False
        else:
            if not isinstance(node_ids_payload, (list, tuple, set)):
                raise ValueError("contribution_node_ids must be a list, tuple, or set.")
            node_ids = [str(item) for item in node_ids_payload]
            require_each_declared = True

        for node_id in node_ids:
            node = graph.nodes.get(node_id)
            if node is None:
                raise ValueError(f"Unknown IPHS contribution node '{node_id}'.")
            node_payload = node.config.get("iphs_contribution")
            if node_payload is None:
                if require_each_declared:
                    raise ValueError(
                        f"Node '{node_id}' does not declare config.iphs_contribution."
                    )
                continue
            if not isinstance(node_payload, Mapping):
                raise ValueError(
                    f"Node '{node_id}' config.iphs_contribution must be a mapping."
                )
            contribution = dict(node_payload)
            declared_source = contribution.get("source_node_id")
            if declared_source is not None and str(declared_source).strip() != node_id:
                raise ValueError(
                    f"Node '{node_id}' IPHS contribution cannot claim source_node_id "
                    f"'{declared_source}'."
                )
            contribution["source_node_id"] = node_id
            contributions.append(contribution)
            contribution_origins.append(f"node_config:{node_id}")
            collected_count += 1

    resolved["contributions"] = tuple(contributions)
    resolved["_compiler_contribution_origins"] = tuple(contribution_origins)
    resolved["_collected_node_contribution_count"] = collected_count
    return resolved


_IPHS_POISSON_FIELD_KEYS = ("poisson_field", "poisson_domain", "jacobi_tol")
_IPHS_STRUCTURE_TOLERANCE_KEYS = ("skew_tol", "psd_tol", "degeneracy_tol")
_IPHS_POISSON_FIELD_OPERATORS = frozenset({"constant_field", "state_field"})
_DOTTED_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
_POISSON_FIELD_REFERENCE = re.compile(rf"({_DOTTED_IDENTIFIER}):({_DOTTED_IDENTIFIER})")


def _declared_iphs_tolerance(name: str, value: Any, requirement: str) -> Any:
    """Accept a declared tolerance only as a number; the spec checks its range.

    An omitted key selects the ParameterManager default inside the IPHS owner;
    a declared null or non-number is rejected here and never replaced by it.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"iphs_assembly {name} must be a {requirement} number; got "
            f"{type(value).__name__}."
        )
    return value


def _resolve_iphs_structure_tolerance_arguments(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Forward declared skew_tol, psd_tol and degeneracy_tol of graph.metadata.iphs_assembly."""

    return {
        key: _declared_iphs_tolerance(key, payload[key], "finite non-negative")
        for key in _IPHS_STRUCTURE_TOLERANCE_KEYS
        if key in payload
    }


def _resolve_poisson_field_reference(reference: Any) -> Any:
    """Resolve an importable 'package.module:attribute' reference to a callable J(x)."""

    if not isinstance(reference, str):
        raise ValueError(
            "iphs_assembly poisson_field must be an importable reference "
            f"'package.module:attribute'; got {type(reference).__name__}."
        )
    match = _POISSON_FIELD_REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError(
            "iphs_assembly poisson_field must be an importable reference "
            f"'package.module:attribute'; got {reference!r}."
        )
    module_name, attribute_path = match.group(1), match.group(2)
    try:
        target: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(
            f"iphs_assembly poisson_field module {module_name!r} cannot be imported: {exc}"
        ) from exc
    for part in attribute_path.split("."):
        if not hasattr(target, part):
            raise ValueError(
                f"iphs_assembly poisson_field reference {reference!r} does not resolve: "
                f"attribute {part!r} is missing."
            )
        target = getattr(target, part)
    if not callable(target):
        raise ValueError(
            f"iphs_assembly poisson_field reference {reference!r} does not resolve to a callable."
        )
    return target


def _poisson_domain_tensor(domain: Any) -> torch.Tensor:
    """Convert a JSON list of numeric state vectors to an (m, n) float64 sample tensor."""

    rows_valid = (
        isinstance(domain, (list, tuple))
        and len(domain) > 0
        and all(isinstance(row, (list, tuple)) and len(row) > 0 for row in domain)
        and len({len(row) for row in domain}) == 1
        and all(
            not isinstance(value, bool) and isinstance(value, (int, float))
            for row in domain
            for value in row
        )
    )
    if not rows_valid:
        raise ValueError(
            "iphs_assembly poisson_domain must be a non-empty list of numeric state "
            "vectors of equal length."
        )
    return torch.tensor(
        [[float(value) for value in row] for row in domain],
        dtype=torch.float64,
    )


def _required_iphs_assembly_field(payload: Mapping[str, Any], field: str) -> Any:
    """Return a policy field the IPHS assembly must declare (no value is filled in).

    ``poisson_operator`` and ``reaction_representation`` are required by
    IPHSStructureSpec / assemble_iphs_structure; a declaration without one
    raises ValueError naming the declaration (graph.metadata.iphs_assembly and
    its name, when declared) and the missing field.
    """
    if field in payload:
        return payload[field]
    name = f" '{payload['name']}'" if "name" in payload else ""
    raise ValueError(
        f"graph.metadata.iphs_assembly{name} is missing the required field '{field}'; "
        "declare it explicitly (IPHSStructureSpec has no default for it)."
    )


def _resolve_iphs_poisson_field_arguments(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the JSON-safe Poisson-field declaration of graph.metadata.iphs_assembly.

    References:
      - Grmela and Oettinger (1997), GENERIC Poisson operator requirements.
      - Marsden and Ratiu (1999), Poisson manifolds, Ch. 10.
    Mathematical expression:
      IPHSStructureSpec evaluates J(x) = poisson_field(x) and checks J^T = -J and the
      Jacobi residual at each poisson_domain sample within jacobi_tol; this helper
      only turns the metadata into those constructor arguments.
    Parameter meanings:
      poisson_field: importable reference 'package.module:attribute' resolved here to
        the callable; poisson_domain: list of numeric state vectors (m rows of n
        values); jacobi_tol: a declared number, forwarded to the spec, which
        requires a finite positive value (an omitted key selects the
        ParameterManager default validation.iphs.jacobi_tolerance; a declared
        null or non-number is rejected here, never replaced by that default).
    Units: the declared state basis of the assembly; no conversion is applied.
    Domain/constraints: the keys apply only to poisson_operator constant_field or
      state_field; any other operator, reference form, domain form, or tolerance
      type is rejected. Shape and finiteness of J(x) at the samples are checked by
      the spec when it is constructed.
    Sign convention: not applicable to a declaration lookup.
    """

    declared = {key: payload[key] for key in _IPHS_POISSON_FIELD_KEYS if key in payload}
    if not declared:
        return {}
    operator = _required_iphs_assembly_field(payload, "poisson_operator")
    if operator not in _IPHS_POISSON_FIELD_OPERATORS:
        raise ValueError(
            f"iphs_assembly {sorted(declared)} apply only to poisson_operator "
            f"constant_field or state_field; got poisson_operator={operator!r}."
        )
    arguments: dict[str, Any] = {}
    if "poisson_field" in declared:
        arguments["poisson_field"] = _resolve_poisson_field_reference(declared["poisson_field"])
    if "poisson_domain" in declared:
        arguments["poisson_domain"] = _poisson_domain_tensor(declared["poisson_domain"])
    if "jacobi_tol" in declared:
        arguments["jacobi_tol"] = _declared_iphs_tolerance(
            "jacobi_tol", declared["jacobi_tol"], "finite positive"
        )
    return arguments


_FORBIDDEN_IPHS_BINDING_FIELDS = {
    "_ASSEMBLED_CONTRIBUTION_BINDING",
    "_compiler_binding",
    "compiler_binding",
    "runtime_bound",
    "runtime_binding",
}


def _iphs_compile_identity_failure(
    reason: str,
    *,
    source_node_ids: Iterable[str] = (),
    bindings: Iterable[_IPHSCompileIdentityBinding] = (),
    detail: str | None = None,
) -> dict[str, Any]:
    binding_items = tuple(bindings)
    failure: dict[str, Any] = {
        "compile_identity_bound": False,
        "process_runtime_replacement_verified": False,
        "evidence_scope": "compile_time_identity_only",
        "binding_count": len(binding_items),
        "source_node_ids": sorted({str(item) for item in source_node_ids}),
        "reason": reason,
        "bindings": [item.to_audit_dict() for item in binding_items],
    }
    if detail is not None:
        failure["detail"] = detail
    return failure


def _bind_iphs_contribution_identities(
    resolved_payload: Mapping[str, Any],
    *,
    graph: GraphIR | None,
    registry: NodeRegistry | None,
    runtime_node_ids: set[str] | None,
    execution_steps: Iterable[ExecutionStep] | None,
) -> dict[str, Any]:
    """Bind contribution provenance to registry-resolved executable steps."""

    contributions = tuple(resolved_payload.get("contributions", ()) or ())
    contribution_origins = tuple(
        resolved_payload.get("_compiler_contribution_origins", ()) or ()
    )
    if not contributions:
        return _iphs_compile_identity_failure("contributions_required")
    if len(contribution_origins) != len(contributions):
        return _iphs_compile_identity_failure("contribution_origin_unavailable")
    source_node_ids: list[str] = []
    normalized: list[Mapping[str, Any]] = []
    for contribution, origin in zip(
        contributions,
        contribution_origins,
        strict=True,
    ):
        if not isinstance(contribution, Mapping):
            return _iphs_compile_identity_failure(
                "mapping_contribution_required",
                source_node_ids=source_node_ids,
            )
        forged_fields = sorted(
            field_name
            for field_name in _FORBIDDEN_IPHS_BINDING_FIELDS
            if field_name in contribution
        )
        if forged_fields:
            return _iphs_compile_identity_failure(
                "forged_evidence_field",
                source_node_ids=source_node_ids,
            )
        source_node_id = str(contribution.get("source_node_id", "")).strip()
        if not source_node_id:
            return _iphs_compile_identity_failure(
                "source_node_id_required",
                source_node_ids=source_node_ids,
            )
        if origin != f"node_config:{source_node_id}":
            return _iphs_compile_identity_failure(
                "inline_metadata_not_compiler_bound",
                source_node_ids=[*source_node_ids, source_node_id],
            )
        source_node_ids.append(source_node_id)
        normalized.append(contribution)

    if len(set(source_node_ids)) != len(source_node_ids):
        return _iphs_compile_identity_failure(
            "duplicate_source_node_id",
            source_node_ids=source_node_ids,
        )

    if graph is None or registry is None or runtime_node_ids is None or execution_steps is None:
        return _iphs_compile_identity_failure(
            "compiler_context_required",
            source_node_ids=source_node_ids,
        )
    step_by_id = {step.node_id: step for step in execution_steps}
    if set(runtime_node_ids) != set(step_by_id):
        return _iphs_compile_identity_failure(
            "execution_set_mismatch",
            source_node_ids=source_node_ids,
        )
    bindings: list[_IPHSCompileIdentityBinding] = []
    for contribution, source_node_id in zip(normalized, source_node_ids, strict=True):
        node = graph.nodes.get(source_node_id)
        if node is None:
            return _iphs_compile_identity_failure(
                "source_node_not_in_graph",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        if source_node_id not in runtime_node_ids or source_node_id not in step_by_id:
            return _iphs_compile_identity_failure(
                "source_node_not_executable",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        definition = registry.get_definition(node.class_type)
        if definition is None:
            return _iphs_compile_identity_failure(
                "node_definition_unresolved",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        if definition.node_type_id != node.class_type:
            return _iphs_compile_identity_failure(
                "node_definition_identity_mismatch",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        runtime_class_declared = definition.metadata.get("runtime_class")
        if not isinstance(runtime_class_declared, str) or not runtime_class_declared.strip():
            return _iphs_compile_identity_failure(
                "runtime_class_unbound",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        # NodeRegistry.resolve_class_type reports an unresolvable runtime
        # class through ImportError (module import), TypeError (attribute is
        # not a class), or ValueError (malformed runtime_class); that is a
        # failed identity binding carrying its cause. Other errors propagate.
        try:
            runtime_class = registry.resolve_class_type(node.class_type)
        except (ImportError, TypeError, ValueError) as exc:
            return _iphs_compile_identity_failure(
                "runtime_class_unresolved",
                source_node_ids=source_node_ids,
                bindings=bindings,
                detail=f"{type(exc).__name__}: {exc}",
            )
        if runtime_class is None:
            return _iphs_compile_identity_failure(
                "runtime_class_unresolved",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        runtime_class_identity = (
            f"{runtime_class.__module__}.{runtime_class.__qualname__}"
        )
        if runtime_class_identity != runtime_class_declared.strip():
            return _iphs_compile_identity_failure(
                "runtime_class_identity_mismatch",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        from relic.core.functor import FunctorBase, IPHSDECFunctor

        if not issubclass(runtime_class, FunctorBase):
            return _iphs_compile_identity_failure(
                "runtime_class_not_functor",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        required_iphs_methods = tuple(
            name
            for name in IPHSDECFunctor.__dict__
            if not name.startswith("_")
        )
        if not callable(getattr(runtime_class, "forward", None)) or any(
            not callable(getattr(runtime_class, name, None))
            for name in required_iphs_methods
        ):
            return _iphs_compile_identity_failure(
                "runtime_class_not_iphs_functor",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        step = step_by_id[source_node_id]
        if step.class_type != node.class_type:
            return _iphs_compile_identity_failure(
                "execution_step_identity_mismatch",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        node_contribution = node.config.get("iphs_contribution")
        step_contribution = step.config.get("iphs_contribution")
        if not isinstance(node_contribution, Mapping) or not isinstance(
            step_contribution,
            Mapping,
        ):
            return _iphs_compile_identity_failure(
                "source_node_contribution_missing",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        contribution_payload = dict(contribution)
        contribution_payload.pop("source_node_id", None)
        if (
            _canonical_sha256(contribution_payload)
            != _canonical_sha256(dict(node_contribution))
            or _canonical_sha256(contribution_payload)
            != _canonical_sha256(dict(step_contribution))
        ):
            return _iphs_compile_identity_failure(
                "contribution_identity_mismatch",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        local_state_vars = tuple(
            str(item)
            for item in contribution.get(
                "state_vars",
                resolved_payload.get("state_vars", ()),
            )
        )
        definition_variables = set(definition.inputs) | set(definition.outputs)
        if not local_state_vars or not set(local_state_vars).issubset(
            definition_variables
        ):
            return _iphs_compile_identity_failure(
                "contribution_state_basis_unbound",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        has_declared_ports = bool(contribution.get("ports")) or contribution.get("G") is not None
        if has_declared_ports:
            return _iphs_compile_identity_failure(
                "port_contract_binding_unavailable",
                source_node_ids=source_node_ids,
                bindings=bindings,
            )
        step_payload = {
            "node_id": step.node_id,
            "class_type": step.class_type,
            "inputs": step.inputs,
            "outputs": step.outputs,
            "config": step.config,
        }
        bindings.append(
            _IPHSCompileIdentityBinding(
                source_node_id=source_node_id,
                node_type_id=definition.node_type_id,
                runtime_class=runtime_class_identity,
                runtime_owner=f"{runtime_class_identity}.forward",
                node_definition_sha256=_canonical_sha256(definition.to_dict()),
                execution_step_sha256=_canonical_sha256(step_payload),
                port_contract_status="not_applicable_no_declared_ports",
            )
        )
    return {
        "compile_identity_bound": True,
        "process_runtime_replacement_verified": False,
        "evidence_scope": "compile_time_identity_only",
        "binding_count": len(bindings),
        "source_node_ids": sorted(source_node_ids),
        "reason": "registry_and_execution_step_bound",
        "bindings": [item.to_audit_dict() for item in bindings],
    }


def _run_iphs_assembly_audit(
    payload: Any,
    graph: GraphIR | None = None,
    *,
    registry: NodeRegistry | None = None,
    runtime_node_ids: set[str] | None = None,
    execution_steps: Iterable[ExecutionStep] | None = None,
) -> dict[str, Any]:
    compile_identity_binding = _iphs_compile_identity_failure("not_evaluated")
    if not isinstance(payload, dict):
        return {
            "enabled": True,
            "passed": False,
            "source": "graph.metadata.iphs_assembly",
            "message": "iphs_assembly must be a mapping.",
            "compile_identity_binding": compile_identity_binding,
        }
    # RELIC's own IPHS operator: an import failure is a defect and propagates.
    from relic.computation.operators.iphs import assemble_iphs_structure

    # Audit boundary: a declaration that the IPHS assembly rejects (TypeError,
    # ValueError, RuntimeError raised by assemble_iphs_structure, its
    # validation, or the strict identity binding) is recorded as a failed
    # audit (passed False), which fails the global compile audit.
    try:
        resolved_payload = _resolve_iphs_assembly_payload(payload, graph)
        graph_metadata = graph.metadata if graph is not None and isinstance(graph.metadata, dict) else {}
        strict_assembly = bool(
            resolved_payload.get("strict", False)
            or graph_metadata.get("phase8_strict")
        )
        required_level, level_issues = _required_iphs_level(
            resolved_payload,
            strict=strict_assembly,
            graph_metadata=graph_metadata,
        )
        if level_issues:
            raise ValueError("; ".join(
                f"{issue.message} {issue.expected}" for issue in level_issues
            ))
        strict_assembly = strict_assembly or (
            required_level is not None and required_level >= 4
        )
        if strict_assembly:
            compile_identity_binding = _bind_iphs_contribution_identities(
                resolved_payload,
                graph=graph,
                registry=registry,
                runtime_node_ids=runtime_node_ids,
                execution_steps=execution_steps,
            )
            if compile_identity_binding.get("compile_identity_bound") is not True:
                detail = compile_identity_binding.get("detail")
                raise ValueError(
                    "Strict IPHS compiler identity binding failed: "
                    f"{compile_identity_binding.get('reason', 'unknown_reason')}"
                    + (f" ({detail})." if detail else ".")
                )
        else:
            compile_identity_binding = _iphs_compile_identity_failure(
                "not_required_non_strict"
            )
        poisson_field_arguments = _resolve_iphs_poisson_field_arguments(resolved_payload)
        tolerance_arguments = _resolve_iphs_structure_tolerance_arguments(resolved_payload)
        spec = assemble_iphs_structure(
            name=str(resolved_payload.get("name", "graph_iphs_assembly")),
            state_vars=tuple(str(item) for item in resolved_payload.get("state_vars", ())),
            energy_functional=str(resolved_payload.get("energy_functional", "declared_energy")),
            entropy_functional=str(resolved_payload.get("entropy_functional", "declared_entropy")),
            contributions=resolved_payload.get("contributions", ()),
            port_interconnection_C=resolved_payload.get("port_interconnection_C"),
            port_dissipation_D=resolved_payload.get("port_dissipation_D"),
            formulation=resolved_payload.get("formulation", "unspecified"),
            poisson_operator=_required_iphs_assembly_field(resolved_payload, "poisson_operator"),
            reaction_representation=_required_iphs_assembly_field(
                resolved_payload, "reaction_representation"
            ),
            reversible_operator_state_dependence=resolved_payload.get(
                "reversible_operator_state_dependence",
                "unspecified",
            ),
            jacobi_identity_evidence=resolved_payload.get(
                "jacobi_identity_evidence",
                "",
            ),
            validate=False,
            strict=strict_assembly,
            required_level=required_level,
            **poisson_field_arguments,
            **tolerance_arguments,
        )
        validation = spec.validate(required_level=required_level)
        return {
            "enabled": True,
            "passed": True,
            "source": "graph.metadata.iphs_assembly",
            "name": spec.name,
            "state_var_count": len(spec.state_vars),
            "port_count": len(spec.ports),
            "has_port_matrix": spec.G is not None,
            "has_stoichiometric_N": spec.stoichiometric_N is not None,
            "required_level": required_level,
            "runtime_contribution_count": spec.contribution_count,
            "runtime_contribution_sources": list(spec.contribution_sources),
            "compile_identity_binding": compile_identity_binding,
            "collected_node_contribution_count": int(
                resolved_payload.get("_collected_node_contribution_count", 0)
            ),
            "validation": validation,
        }
    except (TypeError, ValueError, RuntimeError) as exc:
        return {
            "enabled": True,
            "passed": False,
            "source": "graph.metadata.iphs_assembly",
            "message": str(exc),
            "compile_identity_binding": compile_identity_binding,
        }


def _run_integration_plan_audit(
    payload: Any,
    graph: GraphIR,
    runtime_node_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Rebuild and validate one JSON-safe IntegrationPlan declaration."""
    source = "graph.metadata.integration_plan"
    if not isinstance(payload, Mapping):
        return {
            "enabled": True,
            "passed": False,
            "verified": False,
            "source": source,
            "execution_semantics": "metadata_only",
            "solver_execution_count": 0,
            "message": "integration_plan must be a JSON-safe mapping",
        }
    try:
        from relic.computation.solvers.geometric import IntegrationPlan

        integration_plan = IntegrationPlan.from_dict(payload)
        audit = integration_plan.validate(
            runtime_node_ids=(
                set(graph.nodes)
                if runtime_node_ids is None
                else set(runtime_node_ids)
            )
        )
        audit.update(
            {
                "enabled": True,
                "source": source,
                "execution_semantics": "metadata_only",
                "solver_execution_count": 0,
            }
        )
        return audit
    except (TypeError, ValueError, KeyError) as exc:
        return {
            "enabled": True,
            "passed": False,
            "verified": False,
            "source": source,
            "execution_semantics": "metadata_only",
            "solver_execution_count": 0,
            "message": str(exc),
        }


def _conjoin_iphs_audit_results(
    current: Mapping[str, Any],
    assertion_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve assembly failure when an assertion shares iphs_audit."""

    assembly_result = current.get("assembly_result")
    if not isinstance(assembly_result, Mapping):
        assembly_result = (
            current
            if current.get("source") != "not_declared"
            and current.get("passed") is not None
            else None
        )
    enabled_results = [
        result
        for result in (assembly_result, assertion_result)
        if isinstance(result, Mapping) and bool(result.get("enabled", True))
    ]
    passed_values = [result.get("passed") for result in enabled_results]
    if any(value is False for value in passed_values):
        passed: bool | None = False
    elif passed_values and all(value is True for value in passed_values):
        passed = True
    else:
        passed = None
    combined = dict(current) if isinstance(assembly_result, Mapping) else dict(assertion_result)
    combined.update(
        {
            "enabled": bool(enabled_results),
            "passed": passed,
            "source": "compile_graph.conjunctive_iphs_audit",
            "assembly_result": (
                None if assembly_result is None else dict(assembly_result)
            ),
            "assertion_result": dict(assertion_result),
        }
    )
    return combined


def build_global_structural_audit(
    graph: GraphIR,
    validation: ValidationReport,
    metadata: dict[str, Any] | None = None,
    *,
    registry: NodeRegistry | None = None,
    execution_steps: Iterable[ExecutionStep] | None = None,
) -> dict[str, Any]:
    """Build the unified IPHS-DEC compiler audit extension.

    References:
      - van der Schaft and Maschke (1995), port-Hamiltonian systems.
      - Hairer, Lubich, and Wanner (2006), geometric numerical integration.
    Mathematical expression:
      The compiler audit is a product of structural predicates
      A = prod_i a_i over DEC, Hodge, IPHS, port, reaction, and integration
      checks; no new compiler semantics are introduced.
    Parameter meanings:
      graph: existing GraphIR object compiled by compile_graph.
      validation: existing ValidationReport from validate_graph.
      metadata: compile metadata accumulated by compile_graph.
    """

    graph_metadata = graph.metadata if isinstance(graph.metadata, dict) else {}
    compile_metadata = metadata if isinstance(metadata, dict) else {}
    lowering_metadata = compile_metadata.get("state_boundary_lowering", {})
    lowered_runtime_node_ids: set[str] | None = None
    if isinstance(lowering_metadata, Mapping):
        raw_runtime_node_ids = lowering_metadata.get("executable_node_ids")
        if isinstance(raw_runtime_node_ids, (list, tuple, set)):
            lowered_runtime_node_ids = {
                str(node_id) for node_id in raw_runtime_node_ids
            }
    audit: dict[str, Any] = {
        key: _empty_structural_audit() for key in _STRUCTURAL_AUDIT_KEYS
    }

    structural_audits = graph_metadata.get("structural_audits", {})
    if isinstance(structural_audits, dict):
        for key, value in structural_audits.items():
            if key in audit:
                audit[key] = _normalize_unverified_audit_declaration(
                    value,
                    source="graph.metadata.structural_audits",
                )

    for key in _STRUCTURAL_AUDIT_KEYS:
        if key in graph_metadata:
            audit[key] = _normalize_unverified_audit_declaration(
                graph_metadata[key],
                source="graph.metadata",
            )
        if key in compile_metadata:
            audit[key] = _normalize_audit_payload(compile_metadata[key], source="compile_metadata")

    if "iphs_assembly" in graph_metadata:
        assembly_result = _run_iphs_assembly_audit(
            graph_metadata.get("iphs_assembly"),
            graph,
            registry=registry,
            runtime_node_ids=lowered_runtime_node_ids,
            execution_steps=execution_steps,
        )
        audit["iphs_audit"] = {
            **assembly_result,
            "assembly_result": dict(assembly_result),
            "assertion_result": None,
        }

    integration_plan = graph_metadata.get("integration_plan")
    if integration_plan is not None:
        audit["integration_plan_audit"] = _run_integration_plan_audit(
            integration_plan,
            graph,
            runtime_node_ids=lowered_runtime_node_ids,
        )

    assertion_audit = getattr(validation, "assertion_audit", {}) or {}
    if isinstance(assertion_audit, dict):
        audit_key = assertion_audit.get("audit_key")
        structural_result = assertion_audit.get("structural_result")
        if isinstance(audit_key, str) and audit_key in audit and isinstance(structural_result, dict):
            normalized_result = _normalize_audit_payload(
                structural_result,
                source="validation.assertion",
            )
            audit[audit_key] = (
                _conjoin_iphs_audit_results(audit[audit_key], normalized_result)
                if audit_key == "iphs_audit"
                else normalized_result
            )
        for result in assertion_audit.get("results", []) or []:
            if not isinstance(result, dict):
                continue
            result_audit_key = result.get("audit_key")
            result_structural = result.get("structural_result")
            if (
                isinstance(result_audit_key, str)
                and result_audit_key in audit
                and isinstance(result_structural, dict)
            ):
                normalized_result = _normalize_audit_payload(
                    result_structural,
                    source="validation.assertion",
                )
                audit[result_audit_key] = (
                    _conjoin_iphs_audit_results(
                        audit[result_audit_key],
                        normalized_result,
                    )
                    if result_audit_key == "iphs_audit"
                    else normalized_result
                )

    enabled_count = sum(1 for value in audit.values() if bool(value.get("enabled", False)))
    failed_count = sum(1 for value in audit.values() if value.get("passed") is False)
    audit["global_compile_audit"] = {
        "enabled": True,
        "passed": len(validation.errors) == 0 and failed_count == 0,
        "source": "compile_graph",
        "enabled_structural_audit_count": enabled_count,
        "failed_structural_audit_count": failed_count,
    }
    return audit


def _toposort(node_ids: list[str], links: list[LinkSpec]) -> list[str]:
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for link in links:
        src = link.source.node
        dst = link.target.node
        if src in adjacency and dst in indegree:
            adjacency[src].append(dst)
            indegree[dst] += 1

    queue = [node_id for node_id in node_ids if indegree[node_id] == 0]
    queue.sort()
    order: list[str] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for neighbor in sorted(adjacency.get(current, [])):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
                queue.sort()
    if len(order) != len(node_ids):
        raise ValueError("Graph contains cycles or unresolved dependencies.")
    return order


def _lower_external_forcing_boundaries(
    graph: GraphIR,
) -> tuple[GraphIR, dict[str, Any]]:
    """Lower declarative external-forcing nodes to direct state bindings."""

    required_initial_state_keys = graph.required_initial_state_keys()

    unsupported_state_nodes = sorted(
        (
            node_id,
            node.class_type,
        )
        for node_id, node in graph.nodes.items()
        if node.class_type.startswith("State:")
        and node.class_type != "State:ExternalForcing"
    )
    if unsupported_state_nodes:
        node_id, class_type = unsupported_state_nodes[0]
        raise ValueError(
            "Unsupported executable state node "
            f"'{class_type}' for node '{node_id}'. "
            "Only State:ExternalForcing is a declarative boundary."
        )

    boundary_ids = sorted(
        node_id
        for node_id, node in graph.nodes.items()
        if node.class_type == "State:ExternalForcing"
    )
    boundary_id_set = set(boundary_ids)
    executable_nodes = {
        node_id: NodeSpec(
            class_type=node.class_type,
            inputs=dict(node.inputs),
            outputs=list(node.outputs),
            config=dict(node.config),
        )
        for node_id, node in graph.nodes.items()
        if node_id not in boundary_id_set
    }

    lowered_nodes: list[dict[str, Any]] = []
    lowered_links: list[dict[str, str]] = []
    rewritten_targets: set[tuple[str, str]] = set()

    for boundary_id in boundary_ids:
        boundary = graph.nodes[boundary_id]
        if boundary.inputs or boundary.config:
            raise ValueError(
                f"External forcing boundary '{boundary_id}' must not define "
                "inputs or config."
            )
        state_keys = list(boundary.outputs)
        lowered_nodes.append(
            {
                "node_id": boundary_id,
                "class_type": boundary.class_type,
                "state_keys": sorted(state_keys),
            }
        )

    for link in graph.links:
        if link.target.node in boundary_id_set:
            raise ValueError(
                f"External forcing boundary '{link.target.node}' must not have "
                "incoming links."
            )
        if link.source.node not in boundary_id_set:
            continue

        source_port = link.source.port
        target_port = link.target.port
        boundary = graph.nodes[link.source.node]
        if source_port not in boundary.outputs:
            raise ValueError(
                f"External forcing boundary '{link.source.node}' does not "
                f"declare output '{source_port}'."
            )
        if source_port != target_port:
            raise ValueError(
                f"External forcing output '{source_port}' must bind to the "
                f"same-named target input, not '{target_port}'."
            )
        target_key = (link.target.node, target_port)
        if target_key in rewritten_targets:
            raise ValueError(
                f"Target input '{link.target.node}.{target_port}' has more "
                "than one external forcing source."
            )
        rewritten_targets.add(target_key)

        target_node = executable_nodes[link.target.node]
        expected_binding = {
            "link": [link.source.node, source_port],
        }
        if target_node.inputs.get(target_port) != expected_binding:
            raise ValueError(
                f"Target input '{link.target.node}.{target_port}' does not "
                "match its external forcing link declaration."
            )
        target_node.inputs[target_port] = {"state": source_port}
        lowered_links.append(
            {
                "source_node": link.source.node,
                "source_port": source_port,
                "target_node": link.target.node,
                "target_port": target_port,
                "state_key": source_port,
            }
        )

    for node_id, node in executable_nodes.items():
        for input_name, binding in node.inputs.items():
            if not isinstance(binding, Mapping):
                continue
            raw_link = binding.get("link")
            if (
                isinstance(raw_link, (list, tuple))
                and len(raw_link) == 2
                and str(raw_link[0]) in boundary_id_set
            ):
                raise ValueError(
                    f"Executable input '{node_id}.{input_name}' still "
                    f"references lowered boundary '{raw_link[0]}'."
                )

    executable_links = [
        LinkSpec(
            source=LinkEndpoint(
                node=link.source.node,
                port=link.source.port,
            ),
            target=LinkEndpoint(
                node=link.target.node,
                port=link.target.port,
            ),
        )
        for link in graph.links
        if link.source.node not in boundary_id_set
    ]
    executable_graph = GraphIR(
        schema_version=graph.schema_version,
        nodes=executable_nodes,
        links=executable_links,
        metadata=dict(graph.metadata),
        assertion=dict(graph.assertion),
    )
    lowering_metadata = {
        "lowered_nodes": lowered_nodes,
        "lowered_links": sorted(
            lowered_links,
            key=lambda item: (
                item["source_node"],
                item["source_port"],
                item["target_node"],
                item["target_port"],
            ),
        ),
        "required_initial_state_keys": required_initial_state_keys,
        "executable_node_ids": sorted(executable_nodes),
    }
    return executable_graph, lowering_metadata


def _functor_class_resolution_issues(
    graph: GraphIR,
    registry: NodeRegistry,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for node_id, node in graph.nodes.items():
        if not node.class_type.startswith("Functor:"):
            continue
        resolution_error: Exception | None = None
        # Validation boundary: NodeRegistry.resolve_class_type reports an
        # unresolvable class through ImportError, TypeError, or ValueError;
        # that becomes an unresolved_functor_class issue carrying the cause.
        # Other errors are resolver defects and propagate.
        try:
            resolved = registry.resolve_class_type(node.class_type)
        except (ImportError, TypeError, ValueError) as exc:
            resolved = None
            resolution_error = exc
        if resolved is not None:
            continue
        detail = (
            f" Resolution error: {resolution_error}"
            if resolution_error is not None
            else ""
        )
        issues.append(
            ValidationIssue(
                code="unresolved_functor_class",
                message=(
                    f"Functor class '{node.class_type}' for node '{node_id}' "
                    f"could not be resolved.{detail}"
                ),
                node_id=node_id,
                got=node.class_type,
                hint=(
                    "Use an importable dotted Functor class and register its "
                    "authoritative NodeDefinition before strict compilation."
                ),
            )
        )
    return issues


def _raise_graph_validation_failure(
    errors: list[ValidationIssue],
    *,
    suffix: str = "",
) -> None:
    if not errors:
        return
    message = "Graph validation failed"
    if suffix:
        message += f" {suffix}"
    raise ValueError(
        message + ": " + "; ".join(issue.message for issue in errors)
    )


def _requires_unlowered_adapter(validation: ValidationReport) -> bool:
    return any(
        issue.code == "adapter_required" for issue in validation.warnings
    ) or any(
        issue.code == "unlowered_explicit_adapter_requirement"
        for issue in validation.errors
    )


def _initial_issue_is_covered_by_adapter_insertion(
    issue: ValidationIssue,
    insertions: list[AdapterInsertion],
) -> bool:
    """Return whether lowering will resolve this exact endpoint issue."""

    for insertion in insertions:
        if insertion.adapter_type != "unit_cast":
            continue
        if issue.code == "unit_scale_or_offset_unbound":
            source_is_covered = (
                issue.node_id == insertion.source_node
                and issue.port == insertion.source_port
                and issue.got == insertion.details.get("source_units")
            )
            target_is_covered = (
                issue.node_id == insertion.target_node
                and issue.port == insertion.target_port
                and issue.got == insertion.details.get("target_units")
            )
            if source_is_covered or target_is_covered:
                return True
        if issue.code == "unlowered_explicit_adapter_requirement":
            if (
                issue.node_id == insertion.target_node
                and issue.port == insertion.target_port
                and issue.expected == insertion.adapter_type
            ):
                return True
    return False


def _stable_adapter_identity(insertion: AdapterInsertion) -> tuple[str, str, str]:
    payload = {
        "adapter_type": insertion.adapter_type,
        "source_node": insertion.source_node,
        "source_port": insertion.source_port,
        "target_node": insertion.target_node,
        "target_port": insertion.target_port,
        "details": insertion.details,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (
        f"__auto_adapter_{digest}",
        f"__auto_adapted_{digest}",
        (
            "Functor:relic.orchestration.adapters."
            f"ExecutableAdapterFunctor__{digest}"
        ),
    )


def _normalized_execution_group(node: NodeSpec) -> str:
    raw_group = node.config.get("execution_group")
    if not isinstance(raw_group, str):
        raise ValueError(
            "Automatic adapter lowering requires an explicit execution_group "
            "on both endpoint nodes."
        )
    normalized = raw_group.strip().lower()
    if normalized not in {"fast", "medium", "slow"}:
        raise ValueError(
            "Automatic adapter lowering requires execution_group to be "
            "fast, medium, or slow."
        )
    return normalized


def _require_pure_unit_contract(
    source: PortSpec,
    target: PortSpec,
    *,
    unit_converter: Any,
) -> None:
    if source.context_requirements or target.context_requirements:
        raise ValueError(
            "Automatic unit adapter lowering requires context-free ports."
        )
    if (
        str(source.semantics or "").strip().lower()
        != str(target.semantics or "").strip().lower()
        or source.per_timestep != target.per_timestep
        or source.conservative != target.conservative
    ):
        raise ValueError(
            "Automatic unit adapter cannot transform a non-unit port "
            "contract field."
        )
    probe = deepcopy(source)
    probe.units = target.units
    decision = can_connect(
        probe,
        target,
        unit_converter=unit_converter,
        context=None,
        strict=True,
    )
    if (
        not decision.ok
        or decision.errors
        or decision.warnings
        or decision.requires_adapter
    ):
        raise ValueError(
            "Automatic unit adapter cannot transform a non-unit port "
            "contract field."
        )


def _verify_existing_adapter_definition(
    *,
    registry: NodeRegistry,
    node_type_id: str,
    functor: ExecutableAdapterFunctor,
    source_port: PortSpec,
    target_port: PortSpec,
) -> None:
    definition = registry.get_definition(node_type_id)
    if definition is None:
        return
    if definition.metadata.get("adapter_contract") != functor.contract_config():
        raise ValueError(
            f"Auto-adapter registry collision for '{node_type_id}'."
        )
    if set(definition.inputs) != {source_port.name}:
        raise ValueError(
            f"Auto-adapter input contract drift for '{node_type_id}'."
        )
    if set(definition.outputs) != {target_port.name}:
        raise ValueError(
            f"Auto-adapter output contract drift for '{node_type_id}'."
        )
    if definition.metadata.get("runtime_class") != (
        "relic.orchestration.adapters.ExecutableAdapterFunctor"
    ):
        raise ValueError(
            f"Auto-adapter runtime class drift for '{node_type_id}'."
        )
    registered_source = definition.inputs[source_port.name]
    registered_target = definition.outputs[target_port.name]
    expected_source = deepcopy(source_port)
    expected_source.direction = "in"
    expected_source.write_mode = None
    expected_source.required = True
    expected_target = deepcopy(target_port)
    expected_target.direction = "out"
    if registered_source.to_dict() != expected_source.to_dict():
        raise ValueError(
            f"Auto-adapter source contract drift for '{node_type_id}'."
        )
    if registered_target.to_dict() != expected_target.to_dict():
        raise ValueError(
            f"Auto-adapter target contract drift for '{node_type_id}'."
        )


def _lower_context_free_unit_adapters_impl(
    graph: GraphIR,
    insertions: list[AdapterInsertion],
    *,
    registry: NodeRegistry,
    adapter_registry: AdapterRegistry,
    registered_node_types: list[str],
) -> GraphIR:
    """Lower unambiguous unit conversions through existing adapter owners."""

    # References: CF Metadata Conventions 1.8; Jones (1999).
    # Expression: G' replaces edge u -> v by u -> A -> v, where
    # A(x) = convert(x, source_units, target_units).
    # Parameters: u and v are registered nodes; A is a context-free
    # UnitCastAdapter; source_units and target_units come from PortSpec.
    lowered = GraphIR.from_dict(graph.to_dict())
    by_link = {
        (
            item.source_node,
            item.source_port,
            item.target_node,
            item.target_port,
        ): item
        for item in insertions
    }
    if len(by_link) != len(insertions):
        raise ValueError("Duplicate auto-adapter insertion request.")

    lowered_links: list[LinkSpec] = []
    lowered_records: list[dict[str, Any]] = []
    processed: set[tuple[str, str, str, str]] = set()
    for link in lowered.links:
        key = (
            link.source.node,
            link.source.port,
            link.target.node,
            link.target.port,
        )
        insertion = by_link.get(key)
        if insertion is None:
            lowered_links.append(link)
            continue
        if insertion.adapter_type != "unit_cast":
            raise ValueError(
                "Automatic adapter lowering only supports context-free "
                "unit_cast. Insert other adapter types explicitly."
            )

        source_node = lowered.nodes[insertion.source_node]
        target_node = lowered.nodes[insertion.target_node]
        if source_node.class_type.startswith("State:"):
            raise ValueError(
                "Automatic adapter lowering cannot read a declarative state "
                "boundary as a subprocess output."
            )
        source_group = _normalized_execution_group(source_node)
        target_group = _normalized_execution_group(target_node)
        if source_group != target_group:
            raise ValueError(
                "Automatic adapter lowering cannot cross execution groups."
            )

        source_definition = registry.get_definition(source_node.class_type)
        target_definition = registry.get_definition(target_node.class_type)
        if source_definition is None or target_definition is None:
            raise ValueError(
                "Automatic adapter lowering requires registered endpoint "
                "definitions."
            )
        source_port = source_definition.outputs.get(insertion.source_port)
        target_input = target_definition.inputs.get(insertion.target_port)
        if source_port is None or target_input is None:
            raise ValueError(
                "Automatic adapter lowering cannot resolve endpoint ports."
            )
        if (
            source_port.name != insertion.source_port
            or target_input.name != insertion.target_port
        ):
            raise ValueError(
                "Automatic adapter endpoint keys and PortSpec names differ."
            )
        _require_pure_unit_contract(
            source_port,
            target_input,
            unit_converter=adapter_registry.unit_converter,
        )
        variable_name = source_port.standard_name or "generic"
        # UnitConverter.convert rejects an unsupported pair or a missing
        # context with ValueError; that is the context-free precondition of
        # automatic lowering. Converter infrastructure failures propagate.
        try:
            adapter_registry.unit_converter.convert(
                1.0,
                variable_name=variable_name,
                source_units=str(source_port.units),
                target_units=str(target_input.units),
                context=None,
            )
        except ValueError as exc:
            raise ValueError(
                f"Automatic unit adapter requires undeclared runtime context: {exc}"
            ) from exc

        adapter_node_id, _, adapter_node_type = (
            _stable_adapter_identity(insertion)
        )
        lowered_output = insertion.target_port
        if insertion.source_port == lowered_output:
            raise ValueError(
                "Automatic unit adapter lowering requires distinct source "
                "and target variable names to preserve single-writer state."
            )
        if adapter_node_id in lowered.nodes:
            raise ValueError(
                f"Auto-adapter node collision for '{adapter_node_id}'."
            )
        adapter_config = {
            "variable_name": variable_name,
            "source_units": source_port.units,
            "target_units": target_input.units,
        }
        functor = ExecutableAdapterFunctor(
            name=adapter_node_id,
            adapter_type="unit_cast",
            source_node=insertion.source_node,
            source_port=insertion.source_port,
            target_port=lowered_output,
            config=adapter_config,
        )
        adapter_output_port = deepcopy(target_input)
        adapter_output_port.name = lowered_output
        adapter_output_port.direction = "out"
        existing_definition = registry.get_definition(adapter_node_type)
        if existing_definition is None:
            adapter_registry.register_executable_node(
                node_registry=registry,
                node_type_id=adapter_node_type,
                functor=functor,
                source_port=source_port,
                target_port=adapter_output_port,
            )
            registered_node_types.append(adapter_node_type)
        else:
            _verify_existing_adapter_definition(
                registry=registry,
                node_type_id=adapter_node_type,
                functor=functor,
                source_port=source_port,
                target_port=adapter_output_port,
            )

        target_binding = target_node.inputs.get(insertion.target_port)
        expected_binding = {
            "link": [insertion.source_node, insertion.source_port]
        }
        if target_binding != expected_binding:
            raise ValueError(
                "Automatic adapter lowering found inconsistent target input."
            )
        target_node.inputs[insertion.target_port] = {
            "link": [adapter_node_id, lowered_output]
        }
        step_config = {
            **functor.contract_config(),
            "execution_group": target_group,
        }
        lowered.nodes[adapter_node_id] = NodeSpec(
            class_type=adapter_node_type,
            inputs={
                insertion.source_port: {
                    "link": [
                        insertion.source_node,
                        insertion.source_port,
                    ]
                }
            },
            outputs=[lowered_output],
            config=step_config,
        )
        lowered_links.extend(
            [
                LinkSpec(
                    source=LinkEndpoint(
                        node=insertion.source_node,
                        port=insertion.source_port,
                    ),
                    target=LinkEndpoint(
                        node=adapter_node_id,
                        port=insertion.source_port,
                    ),
                ),
                LinkSpec(
                    source=LinkEndpoint(
                        node=adapter_node_id,
                        port=lowered_output,
                    ),
                    target=LinkEndpoint(
                        node=insertion.target_node,
                        port=insertion.target_port,
                    ),
                ),
            ]
        )
        insertion.details.update(
            {
                "lowered": True,
                "adapter_node_id": adapter_node_id,
                "adapter_node_type": adapter_node_type,
                "lowered_output_port": lowered_output,
                "execution_group": target_group,
            }
        )
        lowered_records.append(
            {
                "adapter_type": insertion.adapter_type,
                "adapter_node_id": adapter_node_id,
                "adapter_node_type": adapter_node_type,
                "source": (
                    f"{insertion.source_node}.{insertion.source_port}"
                ),
                "target": (
                    f"{insertion.target_node}.{insertion.target_port}"
                ),
                "lowered_output_port": lowered_output,
            }
        )
        processed.add(key)

    missing = sorted(set(by_link) - processed)
    if missing:
        raise ValueError(
            f"Auto-adapter requests do not match graph links: {missing!r}"
        )
    lowered.links = lowered_links
    lowered.metadata = {
        **lowered.metadata,
        "adapters": "explicit",
        "auto_adapter_lowering": lowered_records,
    }
    return lowered


def _lower_context_free_unit_adapters(
    graph: GraphIR,
    insertions: list[AdapterInsertion],
    *,
    registry: NodeRegistry,
    adapter_registry: AdapterRegistry,
) -> GraphIR:
    registered_node_types: list[str] = []
    try:
        return _lower_context_free_unit_adapters_impl(
            graph,
            insertions,
            registry=registry,
            adapter_registry=adapter_registry,
            registered_node_types=registered_node_types,
        )
    except (
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        RuntimeError,
        ImportError,
        OSError,
    ):
        for node_type_id in reversed(registered_node_types):
            registry.unregister_definition(node_type_id)
        raise


def _declared_input_projection(
    steps: list[ExecutionStep],
    registry: NodeRegistry,
    required_initial_state_keys: Iterable[str],
) -> dict[str, Any]:
    """Report the state keys each executable node receives from GraphExecutor.

    References: GraphExecutor._declared_state_keys (declared input
      projection); FunctorBase.get_required_inputs (the single
      required-input rule behind NodeDefinition port requiredness).
    Mathematical expression: V_n = K_n; U_n = O_n minus inputs(n); for u in
      U_n the report states whether u is in X and W_u = {m : u in outputs(m)}.
    Parameter meanings: V_n state keys visible to the kernel of node n; K_n
      declared state keys of its plan step; O_n optional input ports of its
      registered NodeDefinition; U_n optional inputs the plan leaves unbound;
      X external state keys of the plan (forcing outputs and declared
      states); W_u nodes whose graph or registered outputs include u, node n
      itself included.
    Units: not applicable (names only).
    Domain/constraints: U_n is a disclosure, not an error. An unbound
      optional input is absent for the kernel even when the graph holds its
      name; binding it (link or state) is the only way to make it an input.
    Sign convention: not applicable.
    """
    from .executor import GraphExecutor

    external = {str(key) for key in required_initial_state_keys}
    definitions = {
        step.node_id: registry.get_definition(step.class_type)
        for step in steps
    }
    writers: dict[str, set[str]] = {}
    for step in steps:
        output_names = {str(name) for name in step.outputs}
        definition = definitions[step.node_id]
        if definition is not None:
            output_names.update(definition.outputs)
        for output_name in output_names:
            writers.setdefault(output_name, set()).add(step.node_id)

    nodes: dict[str, Any] = {}
    for step in steps:
        definition = definitions[step.node_id]
        optional_inputs = (
            set()
            if definition is None
            else {
                name
                for name, port in definition.inputs.items()
                if not port.required
            }
        )
        nodes[step.node_id] = {
            "visible_state_keys": sorted(
                {
                    str(key)
                    for key in GraphExecutor._declared_state_keys(
                        step
                    ).values()
                }
            ),
            "unbound_optional_inputs": {
                name: {
                    "external_state": name in external,
                    "writers": sorted(writers.get(name, ())),
                }
                for name in sorted(optional_inputs - set(step.inputs))
            },
        }
    return {
        "schema_version": "1.0.0",
        "policy": "declared_inputs_only",
        "nodes": nodes,
    }


def _compile_graph_impl(
    graph: GraphIR,
    registry: NodeRegistry,
    adapter_registry: AdapterRegistry | None = None,
    auto_adapt: bool = False,
    strict: bool = True,
    context: ConversionContext | None = None,
    optimization_passes: list[OptimizationPass] | None = None,
    enable_optimization: bool = False,
    include_topology: bool = False,
) -> ExecutionPlan:
    source_graph_sha256 = _canonical_sha256(graph.to_dict())
    source_node_types = _node_types_from_graph(graph)
    source_registry_contract_sha256 = _canonical_sha256(
        _registry_contract_payload(registry, source_node_types)
    )
    adapter_registry = adapter_registry or AdapterRegistry()
    effective_passes: tuple[OptimizationPass, ...] = (
        tuple(
            optimization_passes
            if optimization_passes is not None
            else DEFAULT_OPTIMIZATION_PASSES
        )
        if enable_optimization
        else ()
    )
    requested_pass_contract = (
        None
        if optimization_passes is None
        else [
            _optimization_pass_contract(opt_pass)
            for opt_pass in optimization_passes
        ]
    )
    compiler_options_sha256 = _canonical_sha256(
        {
            "strict": strict,
            "auto_adapt": auto_adapt,
            "enable_optimization": enable_optimization,
            "include_topology": include_topology,
            "conversion_context": context,
            "requested_optimization_passes": requested_pass_contract,
            "effective_optimization_passes": [
                _optimization_pass_contract(opt_pass)
                for opt_pass in effective_passes
            ],
            "adapter_registry": _adapter_registry_contract(
                adapter_registry
            ),
        }
    )
    validation = validate_graph(
        graph=graph,
        registry=registry,
        unit_converter=adapter_registry.unit_converter,
        strict=strict,
        context=context,
        ssot_catalog=getattr(registry, "ssot_catalog", None),
    )
    if strict:
        validation.errors.extend(_functor_class_resolution_issues(graph, registry))

    insertions: list[AdapterInsertion] = []
    if auto_adapt:
        for link in graph.links:
            source_node = graph.nodes[link.source.node]
            target_node = graph.nodes[link.target.node]
            source_def = registry.get_definition(source_node.class_type)
            target_def = registry.get_definition(target_node.class_type)
            if source_def is None or target_def is None:
                continue
            source_port = source_def.outputs.get(link.source.port)
            target_port = target_def.inputs.get(link.target.port)
            if source_port is None or target_port is None:
                continue
            decision = can_connect(
                source_port,
                target_port,
                unit_converter=adapter_registry.unit_converter,
                context=context,
                strict=strict,
            )
            if decision.requires_adapter and decision.adapter_type:
                if any(
                    warn.code == "multiple_adapters_required"
                    for warn in decision.warnings
                ):
                    validation.errors.append(
                        ValidationIssue(
                            code="ambiguous_adapter",
                            message="Multiple adapters required for connection.",
                            node_id=link.target.node,
                            port=link.target.port,
                            hint="Insert adapters explicitly or split conversion steps.",
                        )
                    )
                    continue
                if decision.adapter_type != "unit_cast":
                    validation.errors.append(
                        ValidationIssue(
                            code="unsupported_auto_adapter",
                            message=(
                                "Automatic adapter lowering only supports "
                                "context-free unit_cast. Insert this adapter "
                                "explicitly."
                            ),
                            node_id=link.target.node,
                            port=link.target.port,
                            expected="unit_cast",
                            got=decision.adapter_type,
                        )
                    )
                    continue
                details = {
                    "source_units": source_port.units,
                    "target_units": target_port.units,
                    "standard_name": source_port.standard_name,
                    "source_semantics": source_port.semantics,
                    "target_semantics": target_port.semantics,
                    "target_update_mode": target_port.update_mode,
                    "target_write_mode": target_port.write_mode,
                }
                insertions.append(
                    AdapterInsertion(
                        adapter_type=decision.adapter_type,
                        source_node=link.source.node,
                        source_port=link.source.port,
                        target_node=link.target.node,
                        target_port=link.target.port,
                        details=details,
                    )
                )

    blocking_initial_errors = [
        issue
        for issue in validation.errors
        if not (
            auto_adapt
            and _initial_issue_is_covered_by_adapter_insertion(
                issue,
                insertions,
            )
        )
    ]
    _raise_graph_validation_failure(blocking_initial_errors)
    if strict and not auto_adapt and _requires_unlowered_adapter(validation):
        raise ValueError(
            "Graph validation failed: connection requires an executable "
            "adapter node."
        )
    if auto_adapt and insertions:
        graph = _lower_context_free_unit_adapters(
            graph,
            insertions,
            registry=registry,
            adapter_registry=adapter_registry,
        )
        validation = validate_graph(
            graph=graph,
            registry=registry,
            unit_converter=adapter_registry.unit_converter,
            strict=strict,
            context=None,
            ssot_catalog=getattr(registry, "ssot_catalog", None),
        )
        if strict:
            validation.errors.extend(
                _functor_class_resolution_issues(graph, registry)
            )
        _raise_graph_validation_failure(
            validation.errors,
            suffix="after adapter lowering",
        )
        if strict and _requires_unlowered_adapter(validation):
            raise ValueError(
                "Graph validation failed after adapter lowering: an "
                "executable adapter requirement remains."
            )
    elif auto_adapt and any(
        issue.code == "unlowered_explicit_adapter_requirement"
        for issue in validation.errors
    ):
        raise ValueError(
            "Graph validation failed: explicit adapter requirement was not "
            "lowered."
        )

    if enable_optimization:
        optimized = graph
        for opt_pass in effective_passes:
            optimized = opt_pass.apply(optimized, registry)
        graph = optimized
        validation = validate_graph(
            graph=graph,
            registry=registry,
            unit_converter=adapter_registry.unit_converter,
            strict=strict,
            context=context,
            ssot_catalog=getattr(registry, "ssot_catalog", None),
        )
        if strict:
            validation.errors.extend(
                _functor_class_resolution_issues(graph, registry)
            )
        if validation.errors:
            raise ValueError(
                "Graph validation failed after optimization: "
                + "; ".join(issue.message for issue in validation.errors)
            )

    transformed_graph_sha256 = _canonical_sha256(graph.to_dict())
    transformed_node_types = _node_types_from_graph(graph)
    executable_graph, state_boundary_lowering = (
        _lower_external_forcing_boundaries(graph)
    )
    executable_graph_sha256 = _canonical_sha256(
        executable_graph.to_dict()
    )
    executable_node_types = _node_types_from_graph(executable_graph)
    referenced_node_types = tuple(
        sorted(
            set(source_node_types)
            | set(transformed_node_types)
            | set(executable_node_types)
        )
    )
    order = _toposort(
        sorted(executable_graph.nodes.keys()),
        executable_graph.links,
    )
    steps: list[ExecutionStep] = []
    for node_id in order:
        node = executable_graph.nodes[node_id]
        steps.append(
            ExecutionStep(
                node_id=node_id,
                class_type=node.class_type,
                inputs=dict(node.inputs),
                outputs=list(node.outputs),
                config=dict(node.config),
            )
        )

    compile_metadata: dict[str, Any] = {
        "auto_adapt": auto_adapt,
        "schema_version": graph.schema_version,
        "graph_metadata": dict(graph.metadata),
        "state_boundary_lowering": state_boundary_lowering,
        "input_projection": _declared_input_projection(
            steps,
            registry,
            state_boundary_lowering["required_initial_state_keys"],
        ),
    }
    if insertions:
        compile_metadata["lowered_graph"] = graph.to_dict()

    # Optional topology analysis pass (Patch P1: Algebraic Self-Inspection).
    # References:
    # - Godsil & Royle (2001), Algebraic Graph Theory, Ch. 8
    # - Strang (1988), Four fundamental subspaces
    # Mathematical expression:
    # - B = incidence matrix, rank(B), nullity = n - rank(B),
    #   beta_1 = m - rank(B)
    if include_topology:
        from .topology_algebra import get_topology_signature

        compile_metadata["topology_signature"] = get_topology_signature(
            executable_graph
        )
    audit_summary = build_audit_summary(
        validation,
        metadata=compile_metadata,
    )
    audit_summary.update(
        build_global_structural_audit(
            graph,
            validation,
            compile_metadata,
            registry=registry,
            execution_steps=steps,
        )
    )
    if strict:
        failed_structural_audits = [
            key
            for key, value in audit_summary.items()
            if isinstance(value, dict)
            and bool(value.get("enabled", False))
            and value.get("passed") is False
        ]
        if failed_structural_audits:
            # Each failed audit's own message (e.g. the missing iphs_assembly
            # field) is part of the error, so the cause is visible without a plan.
            details = "; ".join(
                f"{key}: {audit_summary[key]['message']}"
                for key in sorted(failed_structural_audits)
                if audit_summary[key].get("message")
            )
            raise ValueError(
                "Structural audit failed: "
                + ", ".join(sorted(failed_structural_audits))
                + (f" ({details})" if details else "")
            )
    compile_metadata["audit_summary"] = audit_summary

    report = CompileReport(
        validation=validation,
        adapter_insertions=insertions,
        metadata=compile_metadata,
    )
    plan = ExecutionPlan(steps=steps, report=report)
    if strict:
        from .executor import GraphExecutor

        compile_metadata["compile_run_witness"] = GraphExecutor(
            plan,
            registry=registry,
        )._build_compile_run_witness()
    else:
        compile_metadata["compile_run_witness"] = {
            "schema_version": "1.0.0",
            "document_kind": "compile_run_witness",
            "passed": False,
            "execution_performed": False,
            "skipped": True,
            "reason": "strict_compilation_required",
        }
    live_source_registry_sha256 = _canonical_sha256(
        _registry_contract_payload(registry, source_node_types)
    )
    if live_source_registry_sha256 != source_registry_contract_sha256:
        raise RuntimeError(
            "Source registry contract changed during graph compilation."
        )
    registry_contract_sha256 = _canonical_sha256(
        _registry_contract_payload(registry, referenced_node_types)
    )
    plan._compiler_seal = _mint_execution_plan_seal(
        plan,
        source_graph_sha256=source_graph_sha256,
        transformed_graph_sha256=transformed_graph_sha256,
        executable_graph_sha256=executable_graph_sha256,
        source_registry_contract_sha256=(
            source_registry_contract_sha256
        ),
        registry_contract_sha256=registry_contract_sha256,
        compiler_options_sha256=compiler_options_sha256,
        source_node_types=source_node_types,
        referenced_node_types=referenced_node_types,
        registry_bound_node_types=_registry_bound_node_types(
            registry,
            referenced_node_types,
        ),
    )
    verify_execution_plan_integrity(plan, registry)
    return plan


def compile_graph(
    graph: GraphIR,
    registry: NodeRegistry,
    adapter_registry: AdapterRegistry | None = None,
    auto_adapt: bool = False,
    strict: bool = True,
    context: ConversionContext | None = None,
    optimization_passes: list[OptimizationPass] | None = None,
    enable_optimization: bool = False,
    include_topology: bool = False,
) -> ExecutionPlan:
    """Compile transactionally through the canonical GraphIR pass chain."""

    existing_node_types = set(registry.definitions)
    generated_prefix = (
        "Functor:relic.orchestration.adapters."
        "ExecutableAdapterFunctor__"
    )
    try:
        return _compile_graph_impl(
            graph=graph,
            registry=registry,
            adapter_registry=adapter_registry,
            auto_adapt=auto_adapt,
            strict=strict,
            context=context,
            optimization_passes=optimization_passes,
            enable_optimization=enable_optimization,
            include_topology=include_topology,
        )
    except Exception:  # noqa: BLE001
        added_node_types = set(registry.definitions) - existing_node_types
        for node_type_id in sorted(added_node_types):
            if node_type_id.startswith(generated_prefix):
                registry.unregister_definition(node_type_id)
        raise


def _parse_synthesis_groups(
    *,
    base_graph: GraphIR,
    request: Mapping[str, Any],
    registry: NodeRegistry,
) -> list[dict[str, Any]]:
    raw_groups = request.get("choice_groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError(
            "Bounded synthesis request requires non-empty choice_groups."
        )
    parsed_groups: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    seen_node_ids: set[str] = set()
    dependency_node_groups: dict[str, str] = {}
    for raw_group in raw_groups:
        if not isinstance(raw_group, Mapping):
            raise TypeError("Each synthesis choice group must be a mapping.")
        group_id = str(raw_group.get("group_id", "")).strip()
        node_id = str(raw_group.get("node_id", "")).strip()
        if not group_id or not node_id:
            raise ValueError(
                "Each synthesis choice group requires group_id and node_id."
            )
        if group_id in seen_group_ids:
            raise ValueError(f"Duplicate synthesis group_id: {group_id}")
        if node_id in seen_node_ids:
            raise ValueError(
                f"More than one synthesis group targets node '{node_id}'."
            )
        if node_id not in base_graph.nodes:
            raise ValueError(
                f"Synthesis group '{group_id}' targets unknown node "
                f"'{node_id}'."
            )
        seen_group_ids.add(group_id)
        seen_node_ids.add(node_id)

        raw_candidates = raw_group.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(
                f"Synthesis group '{group_id}' requires candidates."
            )
        candidates: list[dict[str, Any]] = []
        seen_candidate_ids: set[str] = set()
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, Mapping):
                raise TypeError(
                    f"Candidates in group '{group_id}' must be mappings."
                )
            candidate_id = str(
                raw_candidate.get("candidate_id", "")
            ).strip()
            node_payload = raw_candidate.get("node")
            if not candidate_id or not isinstance(node_payload, Mapping):
                raise ValueError(
                    f"Candidate in group '{group_id}' requires candidate_id "
                    "and node."
                )
            if candidate_id in seen_candidate_ids:
                raise ValueError(
                    f"Duplicate candidate_id '{candidate_id}' in "
                    f"group '{group_id}'."
                )
            seen_candidate_ids.add(candidate_id)
            if "objectives" in raw_candidate:
                raise ValueError(
                    "Automatic Pareto search is deferred; bounded synthesis "
                    "candidates must not declare objectives."
                )
            raw_external_inputs = raw_candidate.get(
                "required_external_inputs",
                [],
            )
            if not isinstance(raw_external_inputs, list):
                raise TypeError(
                    f"Candidate '{candidate_id}' "
                    "required_external_inputs must be a list."
                )
            required_external_inputs = sorted(
                str(name).strip() for name in raw_external_inputs
            )
            if (
                any(not name for name in required_external_inputs)
                or len(required_external_inputs)
                != len(set(required_external_inputs))
            ):
                raise ValueError(
                    f"Candidate '{candidate_id}' external inputs must be "
                    "unique non-empty names."
                )
            parsed_node = GraphIR.from_dict(
                {
                    "schema_version": base_graph.schema_version,
                    "nodes": {
                        "candidate": dict(node_payload),
                    },
                }
            ).nodes["candidate"]
            if registry.get_definition(parsed_node.class_type) is None:
                raise ValueError(
                    f"Synthesis candidate class '{parsed_node.class_type}' "
                    "is not registered."
                )
            raw_dependency_nodes = raw_candidate.get(
                "dependency_nodes",
                {},
            )
            if not isinstance(raw_dependency_nodes, Mapping):
                raise TypeError(
                    f"Candidate '{candidate_id}' dependency_nodes must be "
                    "a mapping."
                )
            dependency_nodes: dict[str, dict[str, Any]] = {}
            for raw_node_id, raw_node_payload in sorted(
                raw_dependency_nodes.items(),
                key=lambda item: str(item[0]),
            ):
                dependency_node_id = str(raw_node_id).strip()
                if not dependency_node_id:
                    raise ValueError(
                        f"Candidate '{candidate_id}' has an empty "
                        "dependency node id."
                    )
                if dependency_node_id in base_graph.nodes:
                    raise ValueError(
                        f"Candidate dependency node '{dependency_node_id}' "
                        "collides with base graph node."
                    )
                existing_group = dependency_node_groups.get(
                    dependency_node_id
                )
                if (
                    existing_group is not None
                    and existing_group != group_id
                ):
                    raise ValueError(
                        f"Candidate dependency node '{dependency_node_id}' "
                        "is declared by more than one choice group."
                    )
                if not isinstance(raw_node_payload, Mapping):
                    raise TypeError(
                        f"Candidate dependency node '{dependency_node_id}' "
                        "must be a mapping."
                    )
                parsed_dependency = GraphIR.from_dict(
                    {
                        "schema_version": base_graph.schema_version,
                        "nodes": {
                            dependency_node_id: dict(raw_node_payload),
                        },
                    }
                ).nodes[dependency_node_id]
                if (
                    registry.get_definition(
                        parsed_dependency.class_type
                    )
                    is None
                ):
                    raise ValueError(
                        "Synthesis candidate dependency class "
                        f"'{parsed_dependency.class_type}' is not registered."
                    )
                dependency_node_groups[dependency_node_id] = group_id
                dependency_nodes[dependency_node_id] = {
                    "class_type": parsed_dependency.class_type,
                    "inputs": deepcopy(parsed_dependency.inputs),
                    "outputs": list(parsed_dependency.outputs),
                    "config": deepcopy(parsed_dependency.config),
                }
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "node": {
                        "class_type": parsed_node.class_type,
                        "inputs": deepcopy(parsed_node.inputs),
                        "outputs": list(parsed_node.outputs),
                        "config": deepcopy(parsed_node.config),
                    },
                    "dependency_nodes": dependency_nodes,
                    "required_external_inputs": required_external_inputs,
                }
            )
        parsed_groups.append(
            {
                "group_id": group_id,
                "node_id": node_id,
                "candidates": sorted(
                    candidates,
                    key=lambda item: item["candidate_id"],
                ),
            }
        )
    return sorted(parsed_groups, key=lambda item: item["group_id"])


def _selection_id(
    groups: list[dict[str, Any]],
    candidates: tuple[dict[str, Any], ...],
) -> str:
    return "|".join(
        f"{group['group_id']}={candidate['candidate_id']}"
        for group, candidate in zip(groups, candidates, strict=True)
    )


def _synthesis_boundary_issue(
    graph: GraphIR,
    *,
    required_outputs: tuple[str, ...],
    required_inputs: tuple[str, ...],
) -> str | None:
    for output_name in required_outputs:
        writers = [
            node_id
            for node_id, node in graph.nodes.items()
            if output_name in node.outputs
        ]
        if len(writers) != 1:
            return (
                f"Required output '{output_name}' has {len(writers)} "
                "declared writers."
            )
    external_inputs = {
        output_name
        for node in graph.nodes.values()
        if node.class_type == "State:ExternalForcing"
        for output_name in node.outputs
    }
    missing_inputs = sorted(set(required_inputs) - external_inputs)
    if missing_inputs:
        return (
            "Required external inputs are missing: "
            + ", ".join(missing_inputs)
        )
    return None


def _serialize_synthesis_issue(
    issue: ValidationIssue,
) -> dict[str, Any]:
    return {
        "code": str(issue.code),
        "message": str(issue.message),
        "node_id": issue.node_id,
        "port": issue.port,
        "expected": issue.expected,
        "got": issue.got,
        "hint": issue.hint,
    }


def synthesize_bounded_graphs(
    base_graph: GraphIR,
    request: Mapping[str, Any],
    *,
    registry: NodeRegistry,
    adapter_registry: AdapterRegistry | None = None,
    strict: bool = True,
) -> BoundedSynthesisResult:
    """Enumerate a finite registered candidate space through compile_graph."""

    # References: Dechter (2003), Constraint Processing; RELIC GraphIR.
    # Expression: Z = product_g C_g; V = {z in Z: compile_graph(G_z) succeeds}.
    # Parameters: C_g is the finite registered candidate set for group g;
    # max_combinations bounds |Z|; strict enforces canonical graph contracts.
    if not isinstance(request, Mapping):
        raise TypeError("Bounded synthesis request must be a mapping.")
    if request.get("schema_version") != "1.0.0":
        raise ValueError(
            "Bounded synthesis request schema_version must be 1.0.0."
        )
    request_id = str(request.get("request_id", "")).strip()
    if not request_id:
        raise ValueError("Bounded synthesis request requires request_id.")
    raw_limit = request.get("max_combinations")
    if (
        isinstance(raw_limit, bool)
        or not isinstance(raw_limit, int)
        or raw_limit < 1
        or raw_limit > 256
    ):
        raise ValueError(
            "max_combinations must be an integer between 1 and 256."
        )
    raw_auto_adapt = request.get("auto_adapt", False)
    if not isinstance(raw_auto_adapt, bool):
        raise TypeError("auto_adapt must be boolean.")
    if "preferences" in request:
        raise ValueError(
            "Automatic Pareto search is deferred; bounded synthesis must "
            "not declare preferences."
        )
    constraints = request.get("constraints")
    if not isinstance(constraints, Mapping) or (
        constraints.get("strict_unique_writer") is not True
    ):
        raise ValueError(
            "Bounded synthesis requires strict_unique_writer=true."
        )
    groups = _parse_synthesis_groups(
        base_graph=base_graph,
        request=request,
        registry=registry,
    )
    has_conditional_external_inputs = any(
        candidate["required_external_inputs"]
        for group in groups
        for candidate in group["candidates"]
    )
    external_state_node_id = str(
        request.get("external_state_node_id", "")
    ).strip()
    if has_conditional_external_inputs:
        if not external_state_node_id:
            raise ValueError(
                "Conditional external inputs require "
                "external_state_node_id."
            )
        external_node = base_graph.nodes.get(external_state_node_id)
        if (
            external_node is None
            or external_node.class_type != "State:ExternalForcing"
        ):
            raise ValueError(
                "external_state_node_id must identify one base graph "
                "State:ExternalForcing node."
            )
    combination_count = math.prod(
        len(group["candidates"]) for group in groups
    )
    if combination_count > raw_limit:
        raise ValueError(
            f"Candidate space {combination_count} exceeds "
            f"max_combinations {raw_limit}."
        )

    raw_required_outputs = request.get("required_outputs", [])
    raw_required_inputs = request.get("required_inputs", [])
    if not isinstance(raw_required_outputs, list) or not isinstance(
        raw_required_inputs,
        list,
    ):
        raise TypeError("required_outputs and required_inputs must be lists.")
    required_outputs = tuple(
        sorted(str(item).strip() for item in raw_required_outputs)
    )
    required_inputs = tuple(
        sorted(str(item).strip() for item in raw_required_inputs)
    )
    if (
        any(not item for item in required_outputs + required_inputs)
        or len(set(required_outputs)) != len(required_outputs)
        or len(set(required_inputs)) != len(required_inputs)
    ):
        raise ValueError(
            "Required synthesis inputs and outputs must be unique "
            "non-empty names."
        )

    valid_graphs: dict[str, GraphIR] = {}
    invalid_reasons: dict[str, str] = {}
    invalid_issues: dict[str, list[dict[str, Any]]] = {}
    adapter_registry = adapter_registry or AdapterRegistry()
    candidate_sets = [group["candidates"] for group in groups]
    for selected in product(*candidate_sets):
        selection = tuple(selected)
        selection_id = _selection_id(groups, selection)
        payload = deepcopy(base_graph.to_dict())
        payload["links"] = None
        conditional_external_inputs: set[str] = set()
        for group, candidate in zip(groups, selection, strict=True):
            payload["nodes"][group["node_id"]] = deepcopy(
                candidate["node"]
            )
            for dependency_node_id, dependency_node in candidate[
                "dependency_nodes"
            ].items():
                if dependency_node_id in payload["nodes"]:
                    raise ValueError(
                        "Selected candidate dependency node "
                        f"'{dependency_node_id}' collides with the graph."
                    )
                payload["nodes"][dependency_node_id] = deepcopy(
                    dependency_node
                )
            conditional_external_inputs.update(
                candidate["required_external_inputs"]
            )
        if conditional_external_inputs:
            external_payload = payload["nodes"].get(
                external_state_node_id
            )
            if not isinstance(external_payload, dict):
                raise ValueError(
                    "Selected graph external state node is malformed."
                )
            existing_outputs = external_payload.get("outputs")
            if not isinstance(existing_outputs, list):
                raise TypeError(
                    "External state node outputs must be a list."
                )
            external_payload["outputs"] = sorted(
                {
                    *(str(name) for name in existing_outputs),
                    *conditional_external_inputs,
                }
            )
        payload["metadata"] = {
            **dict(payload.get("metadata", {})),
            "graph_construction": "bounded_contract_synthesis",
            "synthesis_request_id": request_id,
            "synthesis_selection_id": selection_id,
        }
        candidate_graph = GraphIR.from_dict(payload)
        boundary_issue = _synthesis_boundary_issue(
            candidate_graph,
            required_outputs=required_outputs,
            required_inputs=required_inputs,
        )
        if boundary_issue is not None:
            invalid_reasons[selection_id] = boundary_issue
            invalid_issues[selection_id] = [
                {
                    "code": "synthesis_boundary_rejected",
                    "message": boundary_issue,
                    "node_id": None,
                    "port": None,
                    "expected": None,
                    "got": None,
                    "hint": None,
                }
            ]
            continue
        # Enumeration boundary: compile_graph rejects an invalid candidate with
        # ValueError (graph validation failure); the candidate is recorded as
        # invalid with its reason and validation report. Any other error is a
        # compiler or registry defect and propagates.
        try:
            compile_graph(
                candidate_graph,
                registry,
                adapter_registry=adapter_registry,
                auto_adapt=raw_auto_adapt,
                strict=strict,
            )
        except ValueError as exc:
            invalid_reasons[selection_id] = str(exc)
            rejection_report = validate_graph(
                graph=candidate_graph,
                registry=registry,
                unit_converter=adapter_registry.unit_converter,
                strict=strict,
                context=None,
                ssot_catalog=getattr(registry, "ssot_catalog", None),
            )
            if strict:
                rejection_report.errors.extend(
                    _functor_class_resolution_issues(
                        candidate_graph,
                        registry,
                    )
                )
            serialized_issues = [
                _serialize_synthesis_issue(issue)
                for issue in rejection_report.errors
            ]
            if not serialized_issues:
                serialized_issues = [
                    {
                        "code": "compile_rejected",
                        "message": str(exc),
                        "node_id": None,
                        "port": None,
                        "expected": None,
                        "got": None,
                        "hint": None,
                    }
                ]
            invalid_issues[selection_id] = sorted(
                serialized_issues,
                key=lambda issue: (
                    str(issue["code"]),
                    str(issue["node_id"]),
                    str(issue["port"]),
                    str(issue["message"]),
                ),
            )
            continue
        valid_graphs[selection_id] = candidate_graph

    return BoundedSynthesisResult(
        request_id=request_id,
        enumerated_count=combination_count,
        valid_graphs=dict(sorted(valid_graphs.items())),
        invalid_reasons=dict(sorted(invalid_reasons.items())),
        invalid_issues=dict(sorted(invalid_issues.items())),
    )


class RELICModelCompiler:
    """Facade over the existing GraphIR compiler with structural audit output."""

    def __init__(
        self,
        registry: NodeRegistry,
        adapter_registry: AdapterRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.adapter_registry = adapter_registry

    def compile(
        self,
        graph: GraphIR,
        **compile_kwargs: Any,
    ) -> ExecutionPlan:
        """Compile through compile_graph and return its existing ExecutionPlan."""

        return compile_graph(
            graph=graph,
            registry=self.registry,
            adapter_registry=self.adapter_registry,
            **compile_kwargs,
        )


def compile_and_execute(
    graph: GraphIR,
    registry: NodeRegistry,
    initial_state: dict[str, Any],
    n_steps: int,
    dt: float,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    cache: Any | None = None,
    simulator_config: Mapping[str, Any] | None = None,
    **compile_kwargs: Any,
) -> dict[str, Any]:
    """Compile a graph and execute it with the core simulator."""
    plan = compile_graph(graph, registry, **compile_kwargs)
    cache_obj = cache
    if cache_obj is None and cache_dir is not None:
        from .cache import NodeCache
        cache_obj = NodeCache(cache_dir)

    executor_module = importlib.import_module("relic.orchestration.executor")
    graph_executor_cls = executor_module.GraphExecutor

    executor = graph_executor_cls(
        plan,
        cache=cache_obj,
        output_dir=output_dir,
        registry=registry,
        simulator_config=simulator_config,
    )
    execution_result = executor.execute(initial_state, n_steps=n_steps, dt=dt)
    if isinstance(execution_result, dict):
        return cast(dict[str, Any], execution_result)
    raise TypeError("GraphExecutor.execute() must return a dictionary state payload")


def plan_to_dict(plan: ExecutionPlan) -> dict[str, Any]:
    """Serialize execution plan to a JSON-friendly dict."""
    seal = _verify_execution_plan_content(plan)
    report_payload: dict[str, Any] = {
        "errors": [issue.__dict__ for issue in plan.report.validation.errors],
        "warnings": [issue.__dict__ for issue in plan.report.validation.warnings],
        "assertion_audit": dict(plan.report.validation.assertion_audit),
        "adapter_insertions": [item.__dict__ for item in plan.report.adapter_insertions],
        "metadata": dict(plan.report.metadata),
        "compiler_seal": seal.to_dict(),
    }
    audit_payload = plan.report.metadata.get("audit_summary")
    if isinstance(audit_payload, dict):
        report_payload["audit"] = dict(audit_payload)
    return {
        "steps": [step.__dict__ for step in plan.steps],
        "report": report_payload,
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)


def write_run_bundle(
    graph: GraphIR,
    plan: ExecutionPlan,
    output_dir: Path,
    *,
    run_id: str | None = None,
    input_files: list[Path] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Write and seal the canonical graph compilation run bundle."""
    if type(graph) is not GraphIR:
        raise TypeError("Run bundle graph must use the canonical GraphIR type.")
    seal = _verify_execution_plan_content(plan)
    if _canonical_sha256(graph.to_dict()) != seal.source_graph_sha256:
        raise ValueError(
            "Run bundle source graph does not match the compiler seal."
        )
    plan_payload = plan_to_dict(plan)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = output_dir / "workflow.json"
    execution_plan_path = output_dir / "execution_plan.json"
    report_path = output_dir / "compile_report.json"
    runtime_path = output_dir / "runtime.log.jsonl"

    _write_json(workflow_path, graph.to_dict())
    _write_json(execution_plan_path, plan_payload["steps"])
    _write_json(report_path, plan_payload["report"])

    declared_inputs: list[Path] = []
    if input_files is not None:
        declared_inputs = [Path(item) for item in input_files]
    else:
        raw_inputs = graph.metadata.get("input_files")
        if raw_inputs is not None:
            if not isinstance(raw_inputs, list):
                raise TypeError("graph.metadata.input_files must be a list")
            for item in raw_inputs:
                if not isinstance(item, (str, Path)):
                    raise TypeError(
                        "graph.metadata.input_files entries must be paths"
                    )
                declared_inputs.append(Path(item))

    bundle_metadata: dict[str, Any] = {
        "schema_version": graph.schema_version,
        "graph_metadata": dict(graph.metadata),
    }
    if metadata is not None:
        caller_metadata = dict(metadata)
        reserved_keys = set(bundle_metadata).intersection(caller_metadata)
        if reserved_keys:
            keys = ", ".join(sorted(reserved_keys))
            raise ValueError(f"Run bundle metadata keys are reserved: {keys}")
        bundle_metadata.update(caller_metadata)

    from ..core.provenance import ProvenanceRecorder

    recorder = ProvenanceRecorder(output_dir=output_dir, run_id=run_id)
    run_bundle = recorder.record_run_bundle(
        artifacts={
            "workflow": workflow_path,
            "execution_plan": execution_plan_path,
            "compile_report": report_path,
            "runtime_log": runtime_path,
        },
        input_files=declared_inputs,
        metadata=bundle_metadata,
    )
    if isinstance(run_bundle, dict):
        normalized = {str(key): str(value) for key, value in run_bundle.items()}
        return normalized
    raise TypeError("ProvenanceRecorder.record_run_bundle() must return a dictionary")
