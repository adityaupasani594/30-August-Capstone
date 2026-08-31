"""
models.py
=========
Pydantic request and response models for the FastAPI endpoints.

These models define the API contract between the Python backend and the
TypeScript frontend. Field names and types MUST remain consistent with
the interface definitions in frontend/src/types/index.ts.
"""

from pydantic import BaseModel, Field
from typing import Dict, List


# ---------------------------------------------------------------------------
# /network response models
# ---------------------------------------------------------------------------

class NodeData(BaseModel):
    """Serialized traffic network node returned by GET /network."""
    id: str
    label: str
    initial_cycle_time: float


class EdgeData(BaseModel):
    """Serialized directed traffic edge returned by GET /network."""
    id: str
    source: str
    target: str
    weight: float
    is_reference: bool


class NetworkResponse(BaseModel):
    """
    Full response schema for GET /network.

    Returns the static graph topology together with the hardcoded STQGCN
    predictions so the frontend can render the 'before' graph on page load.
    """
    nodes: List[NodeData]
    edges: List[EdgeData]
    predictions: Dict[str, float] = Field(
        description="Map of edge ID → predicted congestion in veh/hr"
    )


# ---------------------------------------------------------------------------
# /optimize and /results response models
# ---------------------------------------------------------------------------

class CycleTimeEntry(BaseModel):
    """Before / after signal cycle time for a single node (seconds)."""
    old: float
    new: float


class TableRow(BaseModel):
    """Single row in the before/after congestion comparison table."""
    edge: str
    before: float
    after: float
    difference: float
    status: str   # "Improved" | "Worsened" | "Unchanged" | "No Change"
    is_reference: bool


class OptimizationResponse(BaseModel):
    """
    Full response schema for POST /optimize and GET /results.

    Contains every piece of data needed by the frontend dashboard:
    graph visualizations, table, optimization summary, and fitness chart.
    """
    before: Dict[str, float] = Field(
        description="STQGCN-predicted edge congestion (veh/hr)"
    )
    after: Dict[str, float] = Field(
        description="PSO-optimized edge congestion (veh/hr)"
    )
    cycle_times: Dict[str, CycleTimeEntry] = Field(
        description="Signal cycle times per node before and after optimization"
    )
    fitness_history: List[float] = Field(
        description="Global best fitness value at each PSO iteration"
    )
    initial_fitness: float
    final_fitness: float
    iterations: int
    desired_congestion: float = Field(
        description="Q* — the internally computed target congestion value"
    )
    table: List[TableRow] = Field(
        description="Pre-formatted rows for the congestion comparison table"
    )
    green_times: Dict[str, CycleTimeEntry] = Field(
        default={},
        description="Signal green times per edge before and after optimization"
    )
