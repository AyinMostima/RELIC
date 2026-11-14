"""
Dependency Graph Resolver for Functor Execution Ordering

This module provides automatic dependency resolution and topological sorting
for Functor execution. It ensures that functors are executed in the correct
order based on their dependencies, detects circular dependencies, and provides
visualization capabilities.

References
----------
.. [1] Kahn, A. B. (1962). "Topological sorting of large networks".
       Communications of the ACM.
.. [2] Cormen, T. H., et al. (2009). "Introduction to Algorithms" (3rd ed.).
       MIT Press. Chapter 22: Elementary Graph Algorithms.

Mathematical Framework
----------------------
The dependency graph G = (V, E) where:
- V is the set of functors (vertices)
- E is the set of dependencies (directed edges)
- Edge (u, v) means functor u depends on functor v

Topological ordering ensures: if (u, v) in E, then v appears before u in the ordering
"""

from typing import Dict, List, Set, Optional, Any, Tuple, cast
from collections import defaultdict
import heapq
from dataclasses import dataclass, field
import logging
from enum import Enum

import networkx as nx

HAS_NETWORKX = True

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies between functors."""
    DATA = "data"          # Data flow dependency
    TEMPORAL = "temporal"  # Temporal ordering constraint
    RESOURCE = "resource"  # Resource sharing constraint
    OPTIONAL = "optional"  # Optional dependency (can be ignored if unavailable)


@dataclass
class DependencyEdge:
    """Represents a dependency edge in the graph."""
    source: str              # Functor that has the dependency
    target: str              # Functor being depended upon
    dependency_type: DependencyType = DependencyType.DATA
    weight: float = 1.0      # Priority weight for scheduling
    required: bool = True    # Whether this dependency is required (False for optional)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionNode:
    """Node in the execution graph."""
    functor_id: str
    functor: Any  # The actual functor instance
    dependencies: Set[str] = field(default_factory=set)
    lagged_dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    in_degree: int = 0
    out_degree: int = 0
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


DEPENDENCY_CATEGORIES = ("required", "optional", "lagged")


@dataclass(frozen=True)
class DeclaredDependencies:
    """One Functor's normalized dependency declaration.

    ``required`` and ``optional`` providers are same-step edges (an absent
    optional provider is skipped); ``lagged`` providers are read from state
    without a same-step edge. Names keep their declaration order.
    """

    required: Tuple[str, ...] = ()
    optional: Tuple[str, ...] = ()
    lagged: Tuple[str, ...] = ()

    @property
    def same_step(self) -> Tuple[str, ...]:
        """Required then optional providers (the same-step edge candidates)."""
        return self.required + self.optional

    def canonical(self) -> Dict[str, List[str]]:
        """Sorted form used by provider contract digests and checkpoints."""
        return {
            "required": sorted(self.required),
            "optional": sorted(self.optional),
            "lagged": sorted(self.lagged),
        }


def normalize_dependency_declaration(declaration: Any, *, owner: str) -> DeclaredDependencies:
    """Normalize one ``get_dependencies()`` declaration (the single owner).

    Accepted forms:
        None                      -> no dependencies
        list or tuple of names    -> all names are required
        mapping with keys that are a subset of ``required``, ``optional``,
        ``lagged`` and list or tuple values of names

    Rules: names must be non-empty strings; a repeated name inside one
    category is kept once; a name in more than one category, an unknown
    category key, a non-sequence category value, or any other declaration
    type raises. Merging a class's static declaration with instance-declared
    names is owned by ``FunctorBase._merge_declared_dependencies``; this
    function only validates and normalizes the merged result.

    Raises:
        TypeError: wrong declaration, category value, or name type.
        ValueError: unknown category, empty name, or a name in two categories.
    """

    def _names(values: Any, category: str) -> Tuple[str, ...]:
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                f"{owner}: '{category}' dependencies must be a list or tuple of "
                f"Functor names, got {type(values).__name__}"
            )
        names: List[str] = []
        for value in values:
            if not isinstance(value, str):
                raise TypeError(
                    f"{owner}: '{category}' dependencies contain a non-string name {value!r}"
                )
            if not value.strip():
                raise ValueError(f"{owner}: '{category}' dependencies contain an empty name")
            if value not in names:
                names.append(value)
        return tuple(names)

    if declaration is None:
        return DeclaredDependencies()
    if isinstance(declaration, (list, tuple)):
        return DeclaredDependencies(required=_names(declaration, "required"))
    if not isinstance(declaration, dict):
        raise TypeError(
            f"{owner}: a dependency declaration must be a list, tuple, or mapping of "
            f"{list(DEPENDENCY_CATEGORIES)}, got {type(declaration).__name__}"
        )
    unknown = [key for key in declaration if key not in DEPENDENCY_CATEGORIES]
    if unknown:
        raise ValueError(
            f"{owner}: unknown dependency categories {unknown!r}; expected a subset "
            f"of {list(DEPENDENCY_CATEGORIES)}"
        )
    categories = {
        category: _names(declaration[category], category)
        for category in DEPENDENCY_CATEGORIES
        if category in declaration
    }
    seen: Dict[str, str] = {}
    for category, names in categories.items():
        for name in names:
            if name in seen:
                raise ValueError(
                    f"{owner}: dependency '{name}' is declared in more than one category "
                    f"('{seen[name]}' and '{category}')"
                )
            seen[name] = category
    return DeclaredDependencies(
        required=categories.get("required", ()),
        optional=categories.get("optional", ()),
        lagged=categories.get("lagged", ()),
    )


class CyclicDependencyError(Exception):
    """Raised when a circular dependency is detected."""
    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")


class MissingDependencyError(Exception):
    """Raised when a required dependency cannot be resolved."""
    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        super().__init__(f"Missing required dependency: {source} depends on non-existent {target}")


class DependencyGraphResolver:
    """
    Resolves dependencies and determines execution order for functors.

    This class implements Kahn's algorithm for topological sorting with
    enhancements for handling optional dependencies and dependency levels.
    The levels are reporting candidates; the simulator executes Functors
    sequentially.

    Attributes
    ----------
    nodes : Dict[str, ExecutionNode]
        Map of functor IDs to execution nodes
    edges : List[DependencyEdge]
        List of all dependency edges
    execution_order : Optional[List[str]]
        Cached topological ordering
    """

    def __init__(self, use_networkx: Optional[bool] = None):
        """Initialize an empty dependency graph.

        Parameters
        ----------
        use_networkx : Optional[bool]
            Whether to use networkx DiGraph (defaults to True if available)
        """
        self.nodes: Dict[str, ExecutionNode] = {}
        self.edges: List[DependencyEdge] = []
        self.execution_order: Optional[List[str]] = None
        self._adjacency_list: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_adjacency: Dict[str, Set[str]] = defaultdict(set)
        # Registration order is the fixed total order used as the topological
        # tie-break on both sorting paths (stable topological sort theorem).
        self._registration_index: Dict[str, int] = {}
        self._next_registration_index: int = 0

        # NetworkX graph for advanced features
        self.use_networkx = use_networkx if use_networkx is not None else HAS_NETWORKX
        self._nx_graph: Optional[Any] = nx.DiGraph() if self.use_networkx else None

        # Optional dependencies tracking
        self._optional_deps: Dict[str, Set[str]] = defaultdict(set)
        # Lagged dependencies tracking (recorded but not enforced in the DAG)
        self._lagged_deps: Dict[str, Set[str]] = defaultdict(set)

        # Auto-build dependencies after adding all functors
        self._auto_build_edges = True

    def _resolve_functor_id(self, functor: Any, functor_id: Optional[str]) -> str:
        """Return the explicit identifier or the Functor's own name.

        The identifier keys the registration order and every edge, so it must
        be a declared, non-empty string; an object identity would differ
        between processes and restarts and is never substituted.
        """
        resolved = functor_id if functor_id is not None else getattr(functor, "name", None)
        if not isinstance(resolved, str) or not resolved:
            raise ValueError(
                f"{type(functor).__name__} needs a non-empty string name (or an explicit "
                f"functor_id) to join the dependency graph, got {resolved!r}"
            )
        return resolved

    def _apply_functor_dependencies(
        self,
        node: ExecutionNode,
        functor: Any,
        functor_id: str,
    ) -> None:
        """Populate dependency sets from the normalized functor declaration."""
        if hasattr(functor, "get_dependencies"):
            declared = normalize_dependency_declaration(
                functor.get_dependencies(), owner=functor_id
            )
            node.dependencies = set(declared.same_step)
            if declared.optional:
                self._optional_deps[functor_id].update(declared.optional)
            if declared.lagged:
                self._lagged_deps[functor_id].update(declared.lagged)
                node.lagged_dependencies.update(declared.lagged)
        # get_dependencies() is the only declaration channel (shared with the
        # simulator); optionality is its ``optional`` category, never an
        # ``optional_dependencies`` instance attribute.

    @staticmethod
    def _extract_networkx_metadata_payload(raw_metadata: Any) -> Dict[str, Any]:
        """Convert metadata to the attribute mapping of the networkx mirror node.

        None (no metadata) gives no attributes; a mapping is used as is; an
        object is converted by its own ``to_dict()`` (errors propagate) or its
        attribute dictionary. Any other value is not metadata and raises.
        """
        if raw_metadata is None:
            return {}
        if isinstance(raw_metadata, dict):
            return raw_metadata
        if hasattr(raw_metadata, "to_dict"):
            return cast(Dict[str, Any], raw_metadata.to_dict())
        if hasattr(raw_metadata, "__dict__"):
            return dict(raw_metadata.__dict__)
        raise TypeError(
            f"Functor metadata must be a mapping or a metadata object, got {type(raw_metadata).__name__}"
        )

    def _add_networkx_node(self, functor_id: str, node: ExecutionNode, functor: Any) -> None:
        """Add node to networkx graph when enabled."""
        if self._nx_graph is None:
            return
        metadata_payload = self._extract_networkx_metadata_payload(node.metadata)
        self._nx_graph.add_node(functor_id, functor=functor, **metadata_payload)

    @staticmethod
    def _copy_execution_node(node: ExecutionNode) -> ExecutionNode:
        """Copy mutable graph bookkeeping while retaining the Functor identity."""
        return ExecutionNode(
            functor_id=node.functor_id,
            functor=node.functor,
            dependencies=set(node.dependencies),
            lagged_dependencies=set(node.lagged_dependencies),
            dependents=set(node.dependents),
            in_degree=node.in_degree,
            out_degree=node.out_degree,
            execution_time=node.execution_time,
            metadata=node.metadata,
        )

    def capture_mutation_state(self) -> Dict[str, Any]:
        """Capture the complete in-memory dependency graph transaction state.

        This snapshot belongs to the dependency graph owner. It is not a durable
        simulator checkpoint and intentionally retains the registered Functor
        object identities.
        """
        return {
            "nodes": {
                node_id: self._copy_execution_node(node)
                for node_id, node in self.nodes.items()
            },
            "edges": list(self.edges),
            "execution_order": (
                None if self.execution_order is None else list(self.execution_order)
            ),
            "adjacency_list": {
                node_id: set(targets)
                for node_id, targets in self._adjacency_list.items()
            },
            "reverse_adjacency": {
                node_id: set(sources)
                for node_id, sources in self._reverse_adjacency.items()
            },
            "optional_deps": {
                node_id: set(dependencies)
                for node_id, dependencies in self._optional_deps.items()
            },
            "lagged_deps": {
                node_id: set(dependencies)
                for node_id, dependencies in self._lagged_deps.items()
            },
            "nx_graph": (
                None if self._nx_graph is None else self._nx_graph.copy(as_view=False)
            ),
            "registration_index": dict(self._registration_index),
            "next_registration_index": self._next_registration_index,
        }

    def restore_mutation_state(self, snapshot: Dict[str, Any]) -> None:
        """Restore a state produced by :meth:`capture_mutation_state`."""
        self.nodes = {
            node_id: self._copy_execution_node(node)
            for node_id, node in snapshot["nodes"].items()
        }
        self._registration_index = dict(snapshot["registration_index"])
        self._next_registration_index = int(snapshot["next_registration_index"])
        self.edges = list(snapshot["edges"])
        self.execution_order = (
            None
            if snapshot["execution_order"] is None
            else list(snapshot["execution_order"])
        )
        self._adjacency_list = defaultdict(
            set,
            {
                node_id: set(targets)
                for node_id, targets in snapshot["adjacency_list"].items()
            },
        )
        self._reverse_adjacency = defaultdict(
            set,
            {
                node_id: set(sources)
                for node_id, sources in snapshot["reverse_adjacency"].items()
            },
        )
        self._optional_deps = defaultdict(
            set,
            {
                node_id: set(dependencies)
                for node_id, dependencies in snapshot["optional_deps"].items()
            },
        )
        self._lagged_deps = defaultdict(
            set,
            {
                node_id: set(dependencies)
                for node_id, dependencies in snapshot["lagged_deps"].items()
            },
        )
        stored_graph = snapshot["nx_graph"]
        self._nx_graph = (
            None if stored_graph is None else stored_graph.copy(as_view=False)
        )

    def add_functor(self, functor: Any, functor_id: Optional[str] = None) -> str:
        """
        Add a functor to the dependency graph.

        Parameters
        ----------
        functor : Any
            Functor instance with optional get_dependencies() method
        functor_id : Optional[str]
            Unique identifier for the functor (uses functor.name if not provided)

        Returns
        -------
        str
            The functor ID used in the graph
        """
        functor_id = self._resolve_functor_id(functor, functor_id)
        if functor_id in self.nodes:
            raise ValueError(
                f"Functor '{functor_id}' already exists in dependency graph"
            )

        # Create node
        node = ExecutionNode(
            functor_id=functor_id,
            functor=functor
        )

        self._apply_functor_dependencies(node, functor, functor_id)

        # Add metadata if available
        if hasattr(functor, "get_metadata"):
            node.metadata = functor.get_metadata()

        self.nodes[functor_id] = node
        self._registration_index[functor_id] = self._next_registration_index
        self._next_registration_index += 1
        self.execution_order = None  # Invalidate cache

        self._add_networkx_node(functor_id, node, functor)

        logger.debug(f"Added functor '{functor_id}' with dependencies: {node.dependencies}")

        return functor_id

    def add_edge(self, source: str, target: str,
                 dependency_type: DependencyType = DependencyType.DATA,
                 weight: float = 1.0,
                 required: bool = True) -> None:
        """
        Add a dependency edge between two functors.

        Parameters
        ----------
        source : str
            ID of the functor with the dependency
        target : str
            ID of the functor being depended upon
        dependency_type : DependencyType
            Type of dependency
        weight : float
            Priority weight for scheduling
        required : bool
            Whether this dependency is required (False for optional dependencies)
        """
        edge = DependencyEdge(
            source=source,
            target=target,
            dependency_type=dependency_type,
            weight=weight,
            required=required
        )

        self.edges.append(edge)
        self._adjacency_list[source].add(target)
        self._reverse_adjacency[target].add(source)

        # Track optional dependencies
        if not required or dependency_type == DependencyType.OPTIONAL:
            self._optional_deps[source].add(target)

        # Update node relationships
        if source in self.nodes:
            self.nodes[source].dependencies.add(target)
            self.nodes[source].out_degree += 1

        if target in self.nodes:
            self.nodes[target].dependents.add(source)
            self.nodes[target].in_degree += 1

        # Add to networkx graph if available
        if self._nx_graph is not None:
            self._nx_graph.add_edge(
                target, source,  # NetworkX: target -> source (data flow direction)
                dependency_type=dependency_type.value,
                weight=weight,
                required=required
            )

        self.execution_order = None  # Invalidate cache

        logger.debug(
            "Added edge: %s -> %s (type: %s, required: %s)",
            source,
            target,
            dependency_type.value,
            required,
        )

    def topological_sort(self, handle_optional: bool = True) -> List[str]:
        """
        Perform topological sorting with Kahn's algorithm (registration-order tie-break).

        Parameters
        ----------
        handle_optional : bool
            Whether to handle optional dependencies gracefully (skip missing deps)

        Returns
        -------
        List[str]
            Ordered list of functor IDs in topological order

        Raises
        ------
        CyclicDependencyError
            If a circular dependency is detected
        """
        if self.execution_order is not None and handle_optional:
            return self.execution_order

        # A strict missing-provider request must revalidate a permissive cache.
        self.execution_order = None
        # Build edges automatically based on dependencies
        self._build_edges_from_dependencies(handle_optional=handle_optional)

        # Tie-break contract (stable topological sort, D84): whenever several
        # nodes have no remaining same-step dependency, the one registered first
        # is emitted first.
        order_key = self._registration_order_key

        # One sorting path: Kahn's algorithm over the authoritative adjacency
        # built above. The optional networkx graph is an introspection mirror
        # and is never sorted (a mirror that lost a node or an edge would
        # otherwise shorten or reorder the execution order).
        # References: Kahn (1962), Commun. ACM 5(11):558-562.
        # Mathematical expression: repeatedly emit the ready node (in-degree 0)
        #   of least registration index r(v); decrement the in-degree of its
        #   dependents; nodes left unemitted lie on or behind a cycle.
        # Parameter meanings: in-degree(v) = number of resolved same-step
        #   providers of v; r(v) = registration index (_registration_index).
        # Units: dimensionless counts and indices.
        # Domain/constraints: missing optional and lagged declarations are absent
        #   from this adjacency; a remaining node raises CyclicDependencyError.
        # Sign convention: providers precede consumers in the returned order.
        in_degree = {
            node_id: len(self._adjacency_list[node_id]) for node_id in self.nodes
        }

        # Ready nodes are kept in a min-heap keyed by registration index so the
        # choice among candidates is the fixed total order, not dict insertion.
        ready: List[tuple[int, str]] = [
            (order_key(node_id), node_id) for node_id, degree in in_degree.items() if degree == 0
        ]
        heapq.heapify(ready)
        result = []

        while ready:
            _, current = heapq.heappop(ready)
            result.append(current)

            # Process all dependents of current node
            for dependent in sorted(self.nodes[current].dependents, key=order_key):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    heapq.heappush(ready, (order_key(dependent), dependent))

        # Check for cycles
        if len(result) != len(self.nodes):
            # Find nodes involved in cycle
            remaining = set(self.nodes.keys()) - set(result)
            cycle = self._find_cycle(remaining)
            raise CyclicDependencyError(cycle)

        self.execution_order = result
        logger.info(f"Topological sort complete. Execution order: {result}")

        return result

    def _registration_order_key(self, node_id: str) -> int:
        """Total order used as the topological tie-break (registration sequence)."""
        try:
            return self._registration_index[node_id]
        except KeyError as error:
            raise KeyError(
                f"Node '{node_id}' has no registration index; graph bookkeeping is inconsistent."
            ) from error

    def _build_edges_from_dependencies(self, handle_optional: bool = True) -> None:
        """Build dependency edges automatically from node dependencies.

        Parameters
        ----------
        handle_optional : bool
            Whether to handle optional dependencies gracefully (skip missing deps)
        """
        # Clear existing edges
        self.edges = []
        self._adjacency_list = defaultdict(set)
        self._reverse_adjacency = defaultdict(set)

        # Clear networkx edges if available
        if self._nx_graph is not None:
            self._nx_graph.clear_edges()

        # Reset dependents
        for node in self.nodes.values():
            node.dependents = set()
            node.in_degree = 0
            node.out_degree = 0

        # Build edges from dependencies
        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                # Check if dependency exists
                if dep not in self.nodes:
                    is_optional = dep in self._optional_deps.get(node_id, set())
                    if handle_optional and is_optional:
                        logger.debug(
                            "Skipping optional dependency: %s -> %s (not found)",
                            node_id,
                            dep,
                        )
                        continue
                    message = (
                        f"Missing required dependency: {node_id} -> {dep} (not found)"
                    )
                    logger.error(message)
                    raise MissingDependencyError(node_id, dep)

                # Determine if this is an optional dependency
                required = dep not in self._optional_deps.get(node_id, set())

                # Add edge: node_id depends on dep
                self.add_edge(source=node_id, target=dep, required=required)

    def _find_cycle(self, nodes: Set[str]) -> List[str]:
        """
        Find a cycle in the given set of nodes using DFS.

        Parameters
        ----------
        nodes : Set[str]
            Nodes to search for cycles

        Returns
        -------
        List[str]
            List of nodes forming a cycle
        """
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._adjacency_list[node]:
                if neighbor not in nodes:
                    continue
                if neighbor not in visited:
                    cycle = dfs(neighbor)
                    if cycle:
                        return cycle
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:]

            path.pop()
            rec_stack.remove(node)
            return None

        for node in nodes:
            if node not in visited:
                cycle = dfs(node)
                if cycle:
                    return cycle

        # Every node Kahn could not emit has an unemitted provider, so a cycle
        # exists among them; reaching here means the adjacency is inconsistent.
        raise RuntimeError(
            f"Dependency graph bookkeeping is inconsistent: no cycle among the unsorted nodes {sorted(nodes)}"
        )

    def get_parallel_groups(self) -> List[List[str]]:
        """
        Group functors by dependency level.

        Functors on one level have no same-step edge between them, so they
        are candidates for concurrent execution; the simulator nevertheless
        executes them sequentially. Used for reporting (get_execution_stats).

        Returns
        -------
        List[List[str]]
            List of levels, where each level contains functor IDs without a
            same-step dependency between them
        """
        if self.execution_order is None:
            self.topological_sort()
        execution_order = self.execution_order or []

        levels: Dict[str, int] = {}
        for node_id in execution_order:
            # Find maximum level of dependencies
            max_level = -1
            for dep in self.nodes[node_id].dependencies:
                if dep in levels:
                    max_level = max(max_level, levels[dep])
            levels[node_id] = max_level + 1

        # Group by level
        groups = defaultdict(list)
        for node_id, level in levels.items():
            groups[level].append(node_id)

        return [groups[i] for i in sorted(groups.keys())]

    def validate_dependencies(self) -> List[str]:
        """
        Validate that all declared dependencies exist in the graph.

        Returns
        -------
        List[str]
            List of missing dependencies
        """
        all_nodes = set(self.nodes.keys())
        missing = []

        for node_id, node in self.nodes.items():
            for dep in node.dependencies:
                if dep not in all_nodes:
                    missing.append(f"{node_id} -> {dep}")
                    logger.warning(
                        "Missing dependency: %s depends on non-existent %s",
                        node_id,
                        dep,
                    )

        return missing

    def add_functor_dynamic(
        self,
        functor: Any,
        functor_id: Optional[str] = None,
        rebuild_graph: bool = True,
    ) -> str:
        """
        Dynamically add a functor to the graph and optionally rebuild execution
        order.

        This graph-only operation does not register a Functor with the Simulator,
        execution groups, StateManager, or ParameterManager. Complete runtime hot
        swapping remains unsupported. Any failure restores the exact pre-call graph
        bookkeeping state.

        Parameters
        ----------
        functor : Any
            Functor instance to add
        functor_id : Optional[str]
            Unique identifier for the functor
        rebuild_graph : bool
            Whether to immediately rebuild the dependency graph and execution order

        Returns
        -------
        str
            The functor ID used in the graph

        Examples
        --------
        >>> resolver = DependencyGraphResolver()
        >>> resolver.add_functor(base_functor)
        >>> resolver.topological_sort()
        >>> resolver.add_functor_dynamic(new_functor, rebuild_graph=True)
        """
        snapshot = self.capture_mutation_state()
        fid = self._resolve_functor_id(functor, functor_id)
        try:
            fid = self.add_functor(functor, functor_id)
            if rebuild_graph:
                self.execution_order = None
                self.topological_sort()
                logger.info(f"Successfully added functor '{fid}' dynamically and rebuilt graph")
        except Exception as exc:
            self.restore_mutation_state(snapshot)
            logger.error("Adding functor '%s' dynamically failed: %s", fid, exc)
            raise

        return fid

    def remove_functor(self, functor_id: str, rebuild_graph: bool = True) -> None:
        """
        Remove a functor from the graph dynamically.

        Parameters
        ----------
        functor_id : str
            ID of the functor to remove
        rebuild_graph : bool
            Whether to immediately rebuild the dependency graph

        Raises
        ------
        KeyError
            If the functor_id does not exist in the graph
        MissingDependencyError
            If a remaining functor declares the target as a required dependency.
        """
        if functor_id not in self.nodes:
            raise KeyError(f"Functor '{functor_id}' not found in graph")

        required_consumers = sorted(
            node_id
            for node_id, node in self.nodes.items()
            if node_id != functor_id
            and functor_id in node.dependencies
            and functor_id not in self._optional_deps.get(node_id, set())
        )
        if required_consumers:
            raise MissingDependencyError(required_consumers[0], functor_id)

        del self.nodes[functor_id]
        self._registration_index.pop(functor_id, None)

        if self._nx_graph is not None:
            self._nx_graph.remove_node(functor_id)

        if functor_id in self._optional_deps:
            del self._optional_deps[functor_id]
        if functor_id in self._lagged_deps:
            del self._lagged_deps[functor_id]

        for node in self.nodes.values():
            node.dependents.discard(functor_id)

        self.edges = [e for e in self.edges if e.source != functor_id and e.target != functor_id]

        self.execution_order = None

        if rebuild_graph:
            self.topological_sort()

        logger.info(f"Removed functor '{functor_id}' from graph")

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the dependency graph.

        Returns
        -------
        Dict[str, Any]
            Graph statistics including node count, edge count, etc.
        """
        if not self.execution_order:
            try:
                self.topological_sort()
                has_cycles = False
            except CyclicDependencyError:
                has_cycles = True
        else:
            has_cycles = False

        parallel_groups = self.get_parallel_groups() if not has_cycles else []

        return {
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'has_cycles': has_cycles,
            'max_parallel_width': max(len(g) for g in parallel_groups) if parallel_groups else 0,
            'execution_depth': len(parallel_groups),
            'missing_dependencies': self.validate_dependencies()
        }

    def visualize(self, output_file: Optional[str] = None) -> str:
        """
        Generate a DOT representation of the dependency graph.

        Parameters
        ----------
        output_file : Optional[str]
            Path to save the DOT file

        Returns
        -------
        str
            DOT format string representation of the graph
        """
        # Build edges if not yet built
        if not self.edges and len(self.nodes) > 0:
            self._build_edges_from_dependencies()

        dot_lines = ["digraph DependencyGraph {"]
        dot_lines.append('    rankdir=TB;')
        dot_lines.append('    node [shape=box];')

        # Add nodes
        for node_id, node in self.nodes.items():
            label = f"{node_id}\\n({node.functor.__class__.__name__})"
            dot_lines.append(f'    "{node_id}" [label="{label}"];')

        # Add edges
        for edge in self.edges:
            style = 'dashed' if edge.dependency_type == DependencyType.OPTIONAL else 'solid'
            dot_lines.append(f'    "{edge.target}" -> "{edge.source}" [style={style}];')

        dot_lines.append("}")

        dot_content = '\n'.join(dot_lines)

        if output_file:
            with open(output_file, 'w', encoding="utf-8") as f:
                f.write(dot_content)
            logger.info(f"Dependency graph saved to {output_file}")

        return dot_content
