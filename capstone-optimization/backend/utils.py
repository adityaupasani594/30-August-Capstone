"""
utils.py
========
General utility and formatting helpers for the PSO optimization framework.

Responsibilities:
    - Total volume calculation (used by redistribution model).
    - Formatting before/after congestion into table-ready records.
    - Computing cycle time improvement summaries for the API response.
"""

from typing import Dict, List
from graph import TrafficGraph


def total_non_reference_congestion(
    congestion: Dict[str, float],
    graph: TrafficGraph,
) -> float:
    """
    Compute the total traffic volume across all non-reference edges.

    This conserved quantity ensures the redistribution model in pso.py
    redistributes — rather than creates or destroys — traffic volume.

    Args:
        congestion : Edge congestion dictionary (edge_id → veh/hr).
        graph      : Traffic network graph.

    Returns:
        Total veh/hr summed across all non-reference edges.
    """
    return sum(
        congestion.get(e.id, 0.0)
        for e in graph.get_non_reference_edges()
    )


def format_congestion_table(
    before: Dict[str, float],
    after: Dict[str, float],
    graph: TrafficGraph,
) -> List[dict]:
    """
    Build a table-ready list of dicts comparing before and after congestion.

    Status logic:
        Improved  → after < before by more than 1 veh/hr.
        Worsened  → after > before by more than 1 veh/hr.
        Unchanged → difference within ±1 veh/hr.
        No Change → reference edge (always excluded from optimization).

    Args:
        before : STQGCN-predicted congestion (initial state).
        after  : PSO-optimized congestion (final state).
        graph  : Traffic network graph.

    Returns:
        List of record dicts with keys: edge, before, after, difference, status,
        is_reference.
    """
    rows = []
    for edge in graph.edges:
        eid = edge.id
        q_before = before.get(eid, before.get(eid.replace("→", "->"), 0.0))
        q_after  = after.get(eid, after.get(eid.replace("→", "->"), 0.0))

        diff = q_after - q_before

        if edge.is_reference:
            status = "Distributed" if diff > 0 else "No Change"
        else:
            if diff < -1.0:
                status = "Improved"
            elif diff > 1.0:
                status = "Worsened"
            else:
                status = "Unchanged"

        rows.append({
            "edge": eid,
            "before": round(q_before, 2),
            "after": round(q_after, 2),
            "difference": round(diff, 2),
            "status": status,
            "is_reference": edge.is_reference,
        })
    return rows


def compute_cycle_time_summary(
    old_cycles: Dict[str, float],
    new_cycles: Dict[str, float],
) -> Dict[str, dict]:
    """
    Produce before/after/improvement summary for signal cycle times.

    The improvement percentage uses a sign convention where:
        Positive % → cycle time reduced (less delay per cycle).
        Negative % → cycle time increased (more time to serve congestion).

    Args:
        old_cycles : Initial cycle times per node (seconds).
        new_cycles : PSO-optimized cycle times per node (seconds).

    Returns:
        Dict mapping node IDs → {old, new, improvement_pct}.
    """
    summary: Dict[str, dict] = {}
    for node_id, old in old_cycles.items():
        new = new_cycles.get(node_id, old)
        # Positive improvement_pct means cycle time decreased (less red time)
        improvement_pct = round(((old - new) / old) * 100, 2) if old != 0 else 0.0
        summary[node_id] = {
            "old": round(old, 2),
            "new": round(new, 2),
            "improvement_pct": improvement_pct,
        }
    return summary
