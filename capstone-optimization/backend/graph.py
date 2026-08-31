"""
graph.py
========
Traffic network graph representation for the 5-node demonstration network.

Network Topology:
    Nodes : A, B, C, D, E
    Edges :
        A→B, A→C, A→D   (from hub node A)
        B→D, C→D         (internal convergence)
        D→E              (outgoing from core)
        E→B              (reference edge — fixed at 0 veh/hr)

Design Principles:
    - Node and Edge are plain dataclasses for clarity and testability.
    - TrafficGraph encapsulates all topology queries used by other modules.
    - To support arbitrary networks in production, replace _build() with a
      constructor accepting a generic node/edge configuration dictionary.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from predictions import (
    INITIAL_CYCLE_TIMES, EDGE_WEIGHTS, REFERENCE_EDGE, EDGE_FEATURES, INITIAL_CONGESTION,
    ADITYA_INITIAL_CYCLE_TIMES, ADITYA_EDGE_WEIGHTS, ADITYA_EDGE_FEATURES, ADITYA_INITIAL_CONGESTION
)
from threshold import calculate_threshold


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """
    A traffic network node associated with a signal controller.

    Attributes:
        id                (str)   : Unique node identifier (e.g. 'A').
        label             (str)   : Display label used in visualizations.
        initial_cycle_time(float) : Baseline cycle time (s) before optimization.
    """
    id: str
    label: str
    initial_cycle_time: float

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary for API responses."""
        return {
            "id": self.id,
            "label": self.label,
            "initial_cycle_time": self.initial_cycle_time,
        }


@dataclass
class Edge:
    """
    A directed traffic edge between two network nodes.

    Attributes:
        source      (str)   : Source (upstream) node ID.
        target      (str)   : Target (downstream) node ID.
        weight      (float) : Importance weight in the objective function.
        capacity    (float) : Edge capacity (veh/hr).
        speed       (float) : Edge speed limit (km/h).
        lanes       (int)   : Number of lanes.
        length      (float) : Edge length (km).
        road_type   (str)   : Classification string.
        threshold   (float) : Congestion threshold.
        is_reference(bool)  : If True, edge is excluded from all optimization.
    """
    source: str
    target: str
    weight: float
    capacity: float
    speed: float
    lanes: int
    length: float
    road_type: str
    threshold: float
    is_reference: bool = False

    @property
    def id(self) -> str:
        """Canonical edge identifier (e.g. 'A→B')."""
        return f"{self.source}\u2192{self.target}"

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary for API responses."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "capacity": self.capacity,
            "speed": self.speed,
            "lanes": self.lanes,
            "length": self.length,
            "road_type": self.road_type,
            "threshold": self.threshold,
            "is_reference": self.is_reference,
        }


# ---------------------------------------------------------------------------
# Graph Container
# ---------------------------------------------------------------------------

class TrafficGraph:
    """
    Directed traffic network graph with topology query helpers.

    All modules (objective, signal_controller, pso) interact with the network
    exclusively through this class, ensuring a single source of truth.

    Future Extensibility:
        Replace _build() with a factory method accepting a configuration dict
        to instantiate arbitrary road networks beyond the 5-node demo.
    """

    def __init__(self, network_type: str = "vedant") -> None:
        self.network_type = network_type
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._edge_index: Dict[str, Edge] = {}  # Fast O(1) edge lookup by ID
        self._build()

    def _build(self) -> None:
        """
        Construct the selected network from configurations.
        """
        self.nodes.clear()
        self.edges.clear()
        self._edge_index.clear()

        if self.network_type == "aditya":
            # ------------------------------------------------------------------
            # Build Nodes (Aditya's Network)
            # ------------------------------------------------------------------
            for node_id, cycle_time in ADITYA_INITIAL_CYCLE_TIMES.items():
                self.nodes[node_id] = Node(
                    id=node_id,
                    label=node_id,
                    initial_cycle_time=cycle_time,
                )

            # ------------------------------------------------------------------
            # Build Edges (Aditya's Network)
            # ------------------------------------------------------------------
            edge_definitions = [
                ("A", "B"), ("B", "A"),
                ("A", "C"), ("C", "A"),
                ("A", "D"), ("D", "A"),
                ("B", "C"), ("C", "B"),
                ("B", "D"), ("D", "B"),
                ("C", "D"), ("D", "C"),
            ]

            for src, tgt in edge_definitions:
                edge_id = f"{src}\u2192{tgt}"
                features = ADITYA_EDGE_FEATURES[edge_id]
                
                threshold = calculate_threshold(
                    capacity=features["capacity"],
                    speed=features["speed"],
                    length=features["length"],
                    road_type=features["road_type"],
                    is_reference=False,
                    network_type="aditya",
                )

                edge = Edge(
                    source=src,
                    target=tgt,
                    weight=ADITYA_EDGE_WEIGHTS[edge_id],
                    capacity=features["capacity"],
                    speed=features["speed"],
                    lanes=features["lanes"],
                    length=features["length"],
                    road_type=features["road_type"],
                    threshold=threshold,
                    is_reference=False,
                )
                self.edges.append(edge)
                self._edge_index[edge_id] = edge
        elif self.network_type == "single8":
            cycle_times = {"V1": 55.0, "V2": 75.0, "V3": 70.0, "V4": 60.0, "V5": 80.0, "V6": 75.0, "V7": 60.0, "V8": 55.0}
            edge_data = [
                ("V1", "V2", 4000.0, 70.0, 4, 3.5, 630.0), ("V1", "V4", 3200.0, 60.0, 3, 2.8, 520.0),
                ("V1", "V5", 3000.0, 60.0, 3, 4.5, 650.0), ("V2", "V3", 3500.0, 60.0, 4, 4.0, 602.0),
                ("V2", "V5", 2800.0, 50.0, 3, 1.8, 717.0), ("V3", "V8", 4200.0, 80.0, 4, 2.5, 494.0),
                ("V4", "V5", 2500.0, 50.0, 2, 1.5, 676.0), ("V5", "V6", 3600.0, 50.0, 4, 2.0, 792.0),
                ("V6", "V2", 2400.0, 50.0, 2, 2.2, 379.0), ("V6", "V7", 2800.0, 50.0, 3, 3.0, 586.0),
                ("V7", "V3", 3100.0, 50.0, 3, 2.1, 525.0),
            ]
            self.nodes = {node_id: Node(node_id, node_id, cycle_time) for node_id, cycle_time in cycle_times.items()}
            for src, tgt, capacity, speed, lanes, length, _flow in edge_data:
                edge = Edge(src, tgt, 1.0, capacity, speed, lanes, length, "Arterial", calculate_threshold(capacity, speed, length, "Arterial", network_type="single8"))
                self.edges.append(edge)
                self._edge_index[edge.id] = edge
        else:
            # ------------------------------------------------------------------
            # Build Nodes (Vedant's Network)
            # ------------------------------------------------------------------
            for node_id, cycle_time in INITIAL_CYCLE_TIMES.items():
                self.nodes[node_id] = Node(
                    id=node_id,
                    label=node_id,
                    initial_cycle_time=cycle_time,
                )

            # ------------------------------------------------------------------
            # Build Edges (Vedant's Network)
            # ------------------------------------------------------------------
            edge_definitions = [
                ("A", "B"),   # A→B : internal arterial
                ("B", "A"),   # B→A : anti-parallel
                ("A", "C"),   # A→C : internal arterial
                ("C", "A"),   # C→A : anti-parallel
                ("A", "D"),   # A→D : incoming to core
                ("D", "A"),   # D→A : anti-parallel
                ("B", "D"),   # B→D : internal arterial
                ("D", "B"),   # D→B : anti-parallel
                ("C", "D"),   # C→D : internal arterial
                ("D", "C"),   # D→C : anti-parallel
                ("D", "E"),   # D→E : outgoing from core
                ("E", "D"),   # E→D : anti-parallel
                ("E", "B"),   # E→B : reference edge (fixed at 0 veh/hr)
                ("B", "E"),   # B→E : anti-parallel
            ]

            for src, tgt in edge_definitions:
                edge_id = f"{src}\u2192{tgt}"
                features = EDGE_FEATURES[edge_id]
                
                is_ref = (edge_id == REFERENCE_EDGE)
                threshold = calculate_threshold(
                    capacity=features["capacity"],
                    speed=features["speed"],
                    length=features["length"],
                    road_type=features["road_type"],
                    is_reference=is_ref,
                    network_type="vedant",
                )

                edge = Edge(
                    source=src,
                    target=tgt,
                    weight=EDGE_WEIGHTS[edge_id],
                    capacity=features["capacity"],
                    speed=features["speed"],
                    lanes=features["lanes"],
                    length=features["length"],
                    road_type=features["road_type"],
                    threshold=threshold,
                    is_reference=is_ref,
                )
                self.edges.append(edge)
                self._edge_index[edge_id] = edge

    # ------------------------------------------------------------------
    # Topology Query Methods
    # ------------------------------------------------------------------

    def get_edge(self, edge_id: str) -> Optional[Edge]:
        """Return the edge object for the given canonical ID (e.g. 'A→B')."""
        return self._edge_index.get(edge_id)

    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """Return all edges whose target node is ``node_id``."""
        return [e for e in self.edges if e.target == node_id]

    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Return all edges whose source node is ``node_id``."""
        return [e for e in self.edges if e.source == node_id]

    def get_non_reference_edges(self) -> List[Edge]:
        """Return all edges that are NOT designated as the reference edge."""
        return [e for e in self.edges if not e.is_reference]

    def get_initial_predictions(self) -> Dict[str, float]:
        """Return default predictions for the active network."""
        if self.network_type == "aditya":
            return ADITYA_INITIAL_CONGESTION.copy()
        if self.network_type == "single8":
            return {"V1→V2": 630.0, "V1→V4": 520.0, "V1→V5": 650.0, "V2→V3": 602.0, "V2→V5": 717.0, "V3→V8": 494.0, "V4→V5": 676.0, "V5→V6": 792.0, "V6→V2": 379.0, "V6→V7": 586.0, "V7→V3": 525.0}
        return INITIAL_CONGESTION.copy()

    def get_initial_cycle_times(self) -> Dict[str, float]:
        """Return baseline signal cycle times for the active network."""
        return {node_id: node.initial_cycle_time for node_id, node in self.nodes.items()}

    def update_capacities(self, capacities: Dict[str, float]) -> None:
        """
        Update edge capacities and recalculate thresholds.
        """
        for edge in self.edges:
            if edge.id in capacities:
                edge.capacity = capacities[edge.id]
                edge.threshold = calculate_threshold(
                    capacity=edge.capacity,
                    speed=edge.speed,
                    length=edge.length,
                    road_type=edge.road_type,
                    is_reference=edge.is_reference,
                    network_type=self.network_type,
                )

    def to_dict(self) -> dict:
        """Serialize the full graph topology for API responses."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }
