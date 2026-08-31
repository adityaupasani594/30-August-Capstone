"""
objective.py
============
Evaluates the fitness of a proposed traffic distribution.
"""

from typing import Dict
from graph import TrafficGraph

import numpy as np

def compute_fitness(
    congestion: Dict[str, float], graph: TrafficGraph, threshold_factor: float = 0.60
) -> float:
    """
    Compute fitness as in pso-final.py:
    Fitness = Peak Congestion Percentage + Penalty for exceeding threshold_factor.
    """
    non_ref_edges = graph.get_non_reference_edges()
    if not non_ref_edges:
        return 0.0

    capacities = np.array([e.capacity for e in non_ref_edges], dtype=float)
    flows = np.array([congestion.get(e.id, 0.0) for e in non_ref_edges], dtype=float)

    caps_safe = np.where(capacities > 0, capacities, 1.0)
    occs = flows / caps_safe

    peak_congestion = float(np.max(occs)) * 100.0 if len(occs) > 0 else 0.0
    penalty = float(np.sum(np.maximum(occs - threshold_factor, 0.0)))

    return peak_congestion + penalty