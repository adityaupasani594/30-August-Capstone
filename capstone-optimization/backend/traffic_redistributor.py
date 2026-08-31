"""
traffic_redistributor.py
========================
Deterministic traffic redistribution model that routes vehicle volume
through the network based on signal cycle times while strictly
conserving total vehicles.
"""

from typing import Dict
from graph import TrafficGraph

import numpy as np

def redistribute_traffic_pso(
    initial_congestion: Dict[str, float],
    signal_plan: Dict[str, float]
) -> Dict[str, float]:
    """
    Redistribute traffic based on pso-final.py formula while strictly conserving total vehicles.

    redistributed = base_traffic * (1 - 0.3 * signal_plan)
    redistributed = (redistributed / sum(redistributed)) * total_traffic
    """
    edge_ids = list(initial_congestion.keys())
    base_traffic = np.array([initial_congestion[e] for e in edge_ids], dtype=float)
    total_traffic = np.sum(base_traffic)

    if total_traffic <= 0:
        return initial_congestion.copy()

    plans = np.array([signal_plan.get(e, 0.5) for e in edge_ids], dtype=float)
    redistributed = base_traffic * (1.0 - 0.3 * plans)

    sum_red = np.sum(redistributed)
    if sum_red > 0:
        redistributed = (redistributed / sum_red) * total_traffic
    else:
        redistributed = base_traffic.copy()

    return {edge_id: float(val) for edge_id, val in zip(edge_ids, redistributed)}


def redistribute_traffic(
    graph: TrafficGraph,
    initial_congestion: Dict[str, float],
    cycle_times: Dict[str, float],
    thresholds: Dict[str, float],
    capacities: Dict[str, float],
    signal_plan: Dict[str, float] | None = None
) -> Dict[str, float]:
    """
    Redistribute traffic. If signal_plan is provided, uses pso-final.py vehicle-conserving formula;
    otherwise computes dynamic flow allocation across the graph.
    """
    if signal_plan is not None:
        return redistribute_traffic_pso(initial_congestion, signal_plan)

    GREEN_SPLITS: Dict[str, float] = {
        "A": 0.55,
        "B": 0.55,
        "C": 0.50,
        "D": 0.50,
        "E": 0.45,
    }

    # 1. Map each edge to its downstream edges and calculate split ratios.
    edge_splits = {}
    for edge in graph.edges:
        outgoing = graph.get_outgoing_edges(edge.target)
        if outgoing:
            ratio = 1.0 / len(outgoing)
            edge_splits[edge.id] = [(out_edge.id, ratio) for out_edge in outgoing]
        else:
            edge_splits[edge.id] = []

    # 2. Compute dynamic green splits per incoming approach of each node based on flow ratios
    incoming_by_target: Dict[str, list] = {}
    for edge in graph.edges:
        if edge.target not in incoming_by_target:
            incoming_by_target[edge.target] = []
        incoming_by_target[edge.target].append(edge)

    dynamic_splits: Dict[str, float] = {}
    for node_id, incoming_edges in incoming_by_target.items():
        flow_ratios = {}
        for edge in incoming_edges:
            rt = edge.road_type.lower()
            base_flow = 2200.0 if rt == "expressway" else 1900.0
            sat_flow = base_flow * edge.lanes
            q = initial_congestion.get(edge.id, 0.0)
            flow_ratios[edge.id] = q / sat_flow

        sum_ratios = sum(flow_ratios.values())
        sum_base_splits = sum(GREEN_SPLITS.get(edge.target, 0.50) for edge in incoming_edges)

        for edge in incoming_edges:
            if sum_ratios > 0.0:
                split = sum_base_splits * (flow_ratios[edge.id] / sum_ratios)
            else:
                split = GREEN_SPLITS.get(edge.target, 0.50)
            dynamic_splits[edge.id] = min(0.90, split)

    # 3. Compute excess congestion, discharge capacity, transferable flow,
    # and receiving capacity for all edges.
    excess: Dict[str, float] = {}
    discharge_capacity: Dict[str, float] = {}
    transferable: Dict[str, float] = {}
    receiving: Dict[str, float] = {}

    for edge in graph.edges:
        predicted = initial_congestion.get(edge.id, 0.0)
        threshold = thresholds.get(edge.id, 0.0)

        excess[edge.id] = max(0.0, predicted - threshold)

        rt = edge.road_type.lower()
        base_flow = 2200.0 if rt == "expressway" else 1900.0
        saturation_flow = base_flow * edge.lanes

        ct = max(10.0, cycle_times.get(edge.target, 60.0))

        split = dynamic_splits.get(edge.id, GREEN_SPLITS.get(edge.target, 0.50))
        g_time = split * ct

        lost_time = 3.0
        eff_green_ratio = max(0.05, min(0.90, (g_time - lost_time) / ct))

        cap = capacities.get(edge.id, saturation_flow)
        discharge_capacity[edge.id] = cap * eff_green_ratio

        damping_factor = 0.20 
        transferable[edge.id] = min(excess[edge.id], discharge_capacity[edge.id]) * damping_factor

        receiving[edge.id] = max(0.0, threshold - predicted)

    proposed_inflow = {e.id: 0.0 for e in graph.edges}
    for edge in graph.edges:
        for ds_id, ratio in edge_splits.get(edge.id, []):
            proposed_inflow[ds_id] += transferable[edge.id] * ratio

    inflow_scale = {}
    for edge in graph.edges:
        p_in = proposed_inflow[edge.id]
        r_cap = receiving[edge.id]
        if p_in > r_cap:
            inflow_scale[edge.id] = r_cap / p_in if p_in > 0.0 else 0.0
        else:
            inflow_scale[edge.id] = 1.0

    allocated_matrix = {e.id: {} for e in graph.edges}
    for edge in graph.edges:
        for ds_id, ratio in edge_splits.get(edge.id, []):
            proposed = transferable[edge.id] * ratio
            allocated_matrix[edge.id][ds_id] = proposed * inflow_scale[ds_id]

    q_new: Dict[str, float] = {}
    for edge in graph.edges:
        old_val = initial_congestion.get(edge.id, 0.0)
        leaving = sum(allocated_matrix[edge.id].values())
        entering = 0.0
        for incoming in graph.get_incoming_edges(edge.source):
            entering += allocated_matrix[incoming.id].get(edge.id, 0.0)

        q_new[edge.id] = max(0.0, old_val - leaving + entering)

    return q_new

