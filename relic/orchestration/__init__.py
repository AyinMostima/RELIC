"""
Graph orchestration package for RELIC.
"""

from .graph_ir import GraphIR, NodeSpec, LinkSpec
from .graph_io import load_graph, save_graph
from .port_spec import PortSpec, NodeDefinition
from .ssot_catalog import SSOTCatalog
from .node_registry import NodeRegistry
from .adapters import (
    AdapterRegistry,
    AreaBasisTransformAdapter,
    ExecutableAdapterFunctor,
    SemanticAdapter,
    SurfaceFluxToVolumeSourceAdapter,
    SupportAggregationAdapter,
    TemporalResampleAdapter,
    UnitCastAdapter,
)
from .validator import validate_graph, can_connect, ConnectionDecision, ValidationReport
from .compiler import (
    RELICModelCompiler,
    compile_graph,
    compile_and_execute,
    ExecutionPlan,
    CompileReport,
)
from .cache import NodeCache
from .executor import GraphExecutor

__all__ = [
    "GraphIR",
    "NodeSpec",
    "LinkSpec",
    "load_graph",
    "save_graph",
    "PortSpec",
    "NodeDefinition",
    "SSOTCatalog",
    "NodeRegistry",
    "AdapterRegistry",
    "AreaBasisTransformAdapter",
    "ExecutableAdapterFunctor",
    "SurfaceFluxToVolumeSourceAdapter",
    "SupportAggregationAdapter",
    "UnitCastAdapter",
    "TemporalResampleAdapter",
    "SemanticAdapter",
    "validate_graph",
    "can_connect",
    "ConnectionDecision",
    "ValidationReport",
    "compile_graph",
    "compile_and_execute",
    "RELICModelCompiler",
    "ExecutionPlan",
    "CompileReport",
    "NodeCache",
    "GraphExecutor",
]
