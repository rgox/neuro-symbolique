"""
Symbolic Processing Unit (SPU) Simulator.

This module simulates a specialized symbolic processing unit based on the REASON
architecture. It represents the symbolic/logical component of the neuro-symbolic
architecture.
"""

from typing import Any, Dict, Optional, Set, List, Tuple, Callable
import time
from collections import defaultdict, deque

from nesy.core.memory import UMA, DeviceType, DataType, MemoryBuffer
from nesy.core.telemetry import TelemetryLogger, EventType
from nesy.hal.device import Device, Workload, WorkloadType, ExecutionResult


class Graph:
    """
    Simple graph data structure for symbolic reasoning.

    Represents knowledge as a directed graph where:
    - Nodes are entities/concepts
    - Edges are relations/predicates
    """

    def __init__(self):
        """Initialize empty graph."""
        self.nodes: Set[str] = set()
        self.edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # src -> [(rel, dst)]
        self.reverse_edges: Dict[str, List[Tuple[str, str]]] = defaultdict(list)  # dst -> [(rel, src)]
        self.node_attributes: Dict[str, Dict[str, Any]] = {}
        self.edge_attributes: Dict[Tuple[str, str, str], Dict[str, Any]] = {}  # (src, rel, dst)

    def add_node(self, node: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add a node to the graph."""
        self.nodes.add(node)
        if attributes:
            self.node_attributes[node] = attributes

    def add_edge(
        self,
        src: str,
        relation: str,
        dst: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add an edge to the graph."""
        self.nodes.add(src)
        self.nodes.add(dst)
        self.edges[src].append((relation, dst))
        self.reverse_edges[dst].append((relation, src))

        if attributes:
            self.edge_attributes[(src, relation, dst)] = attributes

    def get_neighbors(self, node: str, relation: Optional[str] = None) -> List[str]:
        """Get neighbors of a node, optionally filtered by relation."""
        if node not in self.edges:
            return []

        neighbors = []
        for rel, dst in self.edges[node]:
            if relation is None or rel == relation:
                neighbors.append(dst)
        return neighbors

    def query(self, pattern: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Query graph with a pattern.

        Args:
            pattern: Query pattern, e.g., {"?x": "type", "relation": "has_part", "?y": None}

        Returns:
            List of bindings satisfying the pattern
        """
        # Simplified query (just matches single edges for MVP)
        results = []

        if "src" in pattern and "relation" in pattern:
            src = pattern["src"]
            relation = pattern["relation"]

            if src in self.edges:
                for rel, dst in self.edges[src]:
                    if rel == relation:
                        results.append({"src": src, "relation": rel, "dst": dst})

        return results


class SPU(Device):
    """
    Symbolic Processing Unit simulator.

    The SPU specializes in:
    - Graph traversal (BFS, DFS, path finding)
    - Logic evaluation (forward/backward chaining)
    - Symbolic query processing
    - Sparse matrix operations

    Based on the REASON architecture concept:
    - Reconfigurable processing elements (PEs)
    - Modes: Symbolic (BCP), Probabilistic (sum-product), SpMSpM (sparse matrix)
    - Optimized for irregular memory access patterns
    - Low-power operation for continuous reasoning
    """

    def __init__(
        self,
        uma: UMA,
        mode: str = "simple",
        graph_cache_size: int = 1000,
        max_depth: int = 10,
        logger: Optional[TelemetryLogger] = None,
    ):
        """
        Initialize SPU simulator.

        Args:
            uma: Unified Memory Architecture
            mode: Reasoning mode ('simple', 'probabilistic', 'logic')
            graph_cache_size: Maximum nodes in graph cache
            max_depth: Maximum traversal depth
            logger: Optional telemetry logger
        """
        super().__init__(DeviceType.SPU, uma, logger)

        self.mode = mode
        self.graph_cache_size = graph_cache_size
        self.max_depth = max_depth

        # Knowledge graphs (can maintain multiple)
        self.graphs: Dict[str, Graph] = {"default": Graph()}

        # Logic rules (simple forward chaining for MVP)
        self.rules: List[Dict[str, Any]] = []

        # Fact base (simple key-value for MVP)
        self.facts: Set[Tuple[str, str, str]] = set()  # (subject, predicate, object)

        # Performance counters
        self.total_queries = 0
        self.total_inferences = 0

        if logger:
            logger.info(
                "SPU initialized",
                event_type=EventType.DEVICE,
                data={
                    "mode": mode,
                    "graph_cache_size": graph_cache_size,
                    "max_depth": max_depth,
                },
            )

    def supports_workload(self, workload_type: WorkloadType) -> bool:
        """Check if SPU supports this workload type."""
        return workload_type in [
            WorkloadType.GRAPH_TRAVERSAL,
            WorkloadType.LOGIC_EVALUATION,
            WorkloadType.SYMBOLIC_QUERY,
        ]

    def get_capabilities(self) -> Dict[str, Any]:
        """Get SPU capabilities."""
        return {
            "mode": self.mode,
            "graph_cache_size": self.graph_cache_size,
            "max_depth": self.max_depth,
            "num_graphs": len(self.graphs),
            "num_facts": len(self.facts),
            "num_rules": len(self.rules),
            "supported_workloads": [
                WorkloadType.GRAPH_TRAVERSAL.value,
                WorkloadType.LOGIC_EVALUATION.value,
                WorkloadType.SYMBOLIC_QUERY.value,
            ],
        }

    def execute(self, workload: Workload) -> ExecutionResult:
        """
        Execute symbolic workload.

        Args:
            workload: Workload containing query, rule, or traversal operation

        Returns:
            ExecutionResult with symbolic outputs

        Expected workload formats:
            GRAPH_TRAVERSAL:
                - operation: "bfs" | "dfs" | "shortest_path"
                - inputs: {"graph": graph_id, "start": node, "goal": node (optional)}

            LOGIC_EVALUATION:
                - operation: "forward_chain" | "query"
                - inputs: {"facts": [...], "rules": [...], "query": ...}

            SYMBOLIC_QUERY:
                - operation: "query"
                - inputs: {"graph": graph_id, "pattern": {...}}
        """
        if not self.supports_workload(workload.workload_type):
            raise NotImplementedError(
                f"SPU does not support {workload.workload_type}"
            )

        start_time = time.time()

        try:
            # Route to appropriate handler
            if workload.workload_type == WorkloadType.GRAPH_TRAVERSAL:
                outputs = self._execute_traversal(workload)
            elif workload.workload_type == WorkloadType.LOGIC_EVALUATION:
                outputs = self._execute_logic(workload)
            elif workload.workload_type == WorkloadType.SYMBOLIC_QUERY:
                outputs = self._execute_query(workload)
            else:
                raise ValueError(f"Unknown workload type: {workload.workload_type}")

            duration_ms = (time.time() - start_time) * 1000

            result = ExecutionResult(
                success=True,
                outputs=outputs,
                duration_ms=duration_ms,
                memory_used_bytes=self._estimate_memory_usage(),
                device_type=self.device_type,
                metadata={
                    "mode": self.mode,
                    "operation": workload.operation,
                },
            )

            self._update_statistics(result)
            self._log_execution(workload, result)

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            if self.logger:
                self.logger.error(
                    f"SPU execution failed: {str(e)}",
                    event_type=EventType.ERROR,
                    data={"workload": workload.workload_type.value},
                )

            return ExecutionResult(
                success=False,
                outputs={},
                duration_ms=duration_ms,
                memory_used_bytes=0,
                device_type=self.device_type,
                metadata={"error": str(e)},
            )

    def _execute_traversal(self, workload: Workload) -> Dict[str, Any]:
        """Execute graph traversal."""
        operation = workload.operation
        inputs = workload.inputs

        graph_id = inputs.get("graph", "default")
        graph = self.graphs.get(graph_id)
        if graph is None:
            raise ValueError(f"Graph '{graph_id}' not found")

        start = inputs.get("start")
        goal = inputs.get("goal")

        if operation == "bfs":
            path = self._bfs(graph, start, goal)
            return {"path": path, "found": goal in path if goal else True}
        elif operation == "dfs":
            path = self._dfs(graph, start, goal)
            return {"path": path, "found": goal in path if goal else True}
        elif operation == "neighbors":
            neighbors = graph.get_neighbors(start)
            return {"neighbors": neighbors}
        else:
            raise ValueError(f"Unknown traversal operation: {operation}")

    def _execute_logic(self, workload: Workload) -> Dict[str, Any]:
        """Execute logic evaluation."""
        operation = workload.operation
        inputs = workload.inputs

        if operation == "add_fact":
            fact = inputs.get("fact")  # (subject, predicate, object)
            self.facts.add(tuple(fact))
            return {"added": True, "total_facts": len(self.facts)}

        elif operation == "query_fact":
            pattern = inputs.get("pattern")
            results = self._query_facts(pattern)
            return {"results": results, "count": len(results)}

        elif operation == "forward_chain":
            # Simple forward chaining (apply rules to derive new facts)
            new_facts = self._forward_chain()
            return {"new_facts": new_facts, "total_facts": len(self.facts)}

        else:
            raise ValueError(f"Unknown logic operation: {operation}")

    def _execute_query(self, workload: Workload) -> Dict[str, Any]:
        """Execute symbolic query."""
        inputs = workload.inputs
        graph_id = inputs.get("graph", "default")
        pattern = inputs.get("pattern")

        graph = self.graphs.get(graph_id)
        if graph is None:
            raise ValueError(f"Graph '{graph_id}' not found")

        results = graph.query(pattern)
        self.total_queries += 1

        return {"results": results, "count": len(results)}

    def _bfs(
        self, graph: Graph, start: str, goal: Optional[str] = None
    ) -> List[str]:
        """Breadth-first search."""
        if start not in graph.nodes:
            return []

        visited = set()
        queue = deque([(start, [start])])

        while queue and len(visited) < self.graph_cache_size:
            node, path = queue.popleft()

            if node in visited:
                continue
            visited.add(node)

            if goal and node == goal:
                return path

            if len(path) >= self.max_depth:
                continue

            for neighbor in graph.get_neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return list(visited) if not goal else []

    def _dfs(
        self, graph: Graph, start: str, goal: Optional[str] = None
    ) -> List[str]:
        """Depth-first search."""
        if start not in graph.nodes:
            return []

        visited = set()
        stack = [(start, [start])]

        while stack and len(visited) < self.graph_cache_size:
            node, path = stack.pop()

            if node in visited:
                continue
            visited.add(node)

            if goal and node == goal:
                return path

            if len(path) >= self.max_depth:
                continue

            for neighbor in graph.get_neighbors(node):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        return list(visited) if not goal else []

    def _query_facts(self, pattern: Tuple) -> List[Tuple]:
        """
        Query facts with pattern matching.

        Pattern can contain None as wildcards.
        Example: ("cup_1", "on", None) matches all facts about what cup_1 is on.
        """
        results = []
        for fact in self.facts:
            if all(
                p is None or p == f
                for p, f in zip(pattern, fact)
            ):
                results.append(fact)
        return results

    def _forward_chain(self) -> List[Tuple]:
        """
        Simple forward chaining inference.

        Applies rules to derive new facts (MVP version).
        """
        new_facts = []
        # Placeholder for rule-based inference
        # Would iterate through rules and apply them to facts
        return new_facts

    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage of symbolic structures."""
        # Rough estimate: nodes + edges + facts
        memory = 0
        for graph in self.graphs.values():
            memory += len(graph.nodes) * 100  # ~100 bytes per node
            memory += sum(len(edges) for edges in graph.edges.values()) * 150  # ~150 bytes per edge

        memory += len(self.facts) * 200  # ~200 bytes per fact
        return memory

    def add_graph(self, graph_id: str, graph: Optional[Graph] = None) -> None:
        """Add or replace a graph."""
        self.graphs[graph_id] = graph or Graph()

    def get_graph(self, graph_id: str = "default") -> Optional[Graph]:
        """Get a graph by ID."""
        return self.graphs.get(graph_id)

    def add_rule(self, rule: Dict[str, Any]) -> None:
        """Add a logic rule."""
        self.rules.append(rule)

    def clear_facts(self) -> None:
        """Clear all facts."""
        self.facts.clear()

    def clear_graphs(self) -> None:
        """Clear all graphs."""
        self.graphs = {"default": Graph()}
