"""
compare_pso_qaoa.py
===================
Direct Comparison between Frontend PSO (pso.py) and QAOA (final_qaoa_with_signal.py)

Uses the EXACT, UNTOUCHED PSO engine logic from `pso.py` as executed in the web app:
- PSO Engine: `pso.PSO.optimize()` on `TrafficGraph`
- QAOA Engine: `final_qaoa_with_signal.run_final_qaoa_optimization()` (COBYLA maxiter = 50, max 10 packets)

Compares:
  1. Edge Congestion / Traffic Flows (Initial vs PSO vs QAOA)
  2. Edge Occupancy Rates (%) & Peak Reductions
  3. Intersection Cycle Times & Green Light Allocations
  4. Overall System Metrics (Peak reduction %, Load distribution improvement %, Latency ms)
"""

import sys
import time
import numpy as np
from typing import Dict, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


from graph import TrafficGraph
from pso import PSO
from predictions import (
    INITIAL_CONGESTION,
    INITIAL_CYCLE_TIMES,
    ADITYA_INITIAL_CONGESTION,
    ADITYA_INITIAL_CYCLE_TIMES,
)
from final_qaoa_with_signal import run_final_qaoa_optimization
from app import _compute_green_times, _build_response


def run_comparison(network_type: str = "aditya"):
    """
    Executes comparison using the EXACT PSO engine from pso.py and QAOA engine.
    """
    graph = TrafficGraph(network_type=network_type)
    num_nodes = len(graph.nodes)
    print("=" * 95)
    print(f"      FRONTEND COMPARISON: PSO (pso.py) vs QAOA ({num_nodes}-NODE {network_type.upper()} NETWORK)")
    print(f"      (PSO logic: Untouched pso.py | QAOA logic: COBYLA maxiter=50, max 10 packets)")
    print("=" * 95)

    init_congestion = ADITYA_INITIAL_CONGESTION.copy() if network_type == "aditya" else INITIAL_CONGESTION.copy()
    init_cycle_times = ADITYA_INITIAL_CYCLE_TIMES.copy() if network_type == "aditya" else INITIAL_CYCLE_TIMES.copy()
    capacities = {e.id: e.capacity for e in graph.edges}

    # ------------------------------------------------------------------
    # 1. Execute EXACT Frontend PSO Engine (pso.py)
    # ------------------------------------------------------------------
    start_pso = time.perf_counter()
    pso_engine = PSO(max_iter=20)
    pso_raw_res = pso_engine.optimize(
        graph=graph,
        initial_congestion=init_congestion,
        initial_cycle_times=init_cycle_times,
    )
    pso_latency = (time.perf_counter() - start_pso) * 1000.0

    pso_api_res = _build_response(pso_raw_res, graph, init_congestion)
    pso_flows = pso_api_res["optimized_congestion"]
    pso_cycles = pso_api_res["optimized_cycle_times"]
    pso_greens = {e: pso_api_res["green_times"][e]["new"] for e in pso_api_res["green_times"]}

    # ------------------------------------------------------------------
    # 2. Execute QAOA Engine (final_qaoa_with_signal.py)
    # ------------------------------------------------------------------
    start_qaoa = time.perf_counter()
    qaoa_raw_res = run_final_qaoa_optimization(
        input_predicted_loads=init_congestion,
        input_capacities=capacities,
    )
    qaoa_latency = (time.perf_counter() - start_qaoa) * 1000.0

    qaoa_flows = qaoa_raw_res["optimized_congestion"]
    qaoa_cycles = qaoa_raw_res["cycle_times"]
    qaoa_greens = qaoa_raw_res["green_times"]

    def get_flow(flow_dict, edge_id, default_val=0.0):
        if edge_id in flow_dict:
            return flow_dict[edge_id]
        alt_id = edge_id.replace("->", "\u2192") if "->" in edge_id else edge_id.replace("\u2192", "->")
        if alt_id in flow_dict:
            return flow_dict[alt_id]
        return default_val

    # Occupancies
    init_occ = {e.id: get_flow(init_congestion, e.id, 0.0) / e.capacity for e in graph.edges}
    pso_occ = {e.id: get_flow(pso_flows, e.id, 0.0) / e.capacity for e in graph.edges}
    qaoa_occ = {e.id: get_flow(qaoa_flows, e.id, get_flow(init_congestion, e.id, 0.0)) / e.capacity for e in graph.edges}

    # ------------------------------------------------------------------
    # Table 1: Edge Congestion & Occupancy Comparison
    # ------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" 1. EDGE CONGESTION & OCCUPANCY COMPARISON")
    print("-" * 95)
    print(f"{'Edge':<10} | {'Capacity':<10} | {'Initial Flow (Occ%)':<22} | {'PSO Flow (Occ%)':<22} | {'QAOA Flow (Occ%)':<22}")
    print("-" * 95)

    for edge in graph.edges:
        e_id = edge.id
        cap = edge.capacity
        i_f = init_congestion.get(e_id, 0.0)
        i_o = init_occ.get(e_id, 0.0) * 100.0

        p_f = pso_flows.get(e_id, 0.0)
        p_o = pso_occ.get(e_id, 0.0) * 100.0

        q_f = qaoa_flows.get(e_id, 0.0)
        q_o = qaoa_occ.get(e_id, 0.0) * 100.0

        i_str = f"{i_f:.0f} ({i_o:.1f}%)"
        p_str = f"{p_f:.0f} ({p_o:.1f}%)"
        q_str = f"{q_f:.0f} ({q_o:.1f}%)"

        print(f"{e_id:<10} | {cap:<10.0f} | {i_str:<22} | {p_str:<22} | {q_str:<22}")

    print("-" * 95)

    # ------------------------------------------------------------------
    # Table 2: Intersection Cycle Times Comparison
    # ------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" 2. INTERSECTION CYCLE TIMES COMPARISON (seconds)")
    print("-" * 95)
    print(f"{'Intersection':<16} | {'Initial Cycle (s)':<20} | {'PSO Cycle (pso.py)':<25} | {'QAOA Cycle (s)':<20}")
    print("-" * 95)

    all_nodes = sorted(list(graph.nodes.keys()))
    for node in all_nodes:
        i_ct = init_cycle_times.get(node, 50.0)
        p_ct = pso_cycles.get(node, i_ct)
        q_ct = qaoa_cycles.get(node, 90.0)

        print(f"{'Node ' + node:<16} | {i_ct:<20.1f} | {p_ct:<25.1f} | {q_ct:<20.1f}")

    print("-" * 95)

    # ------------------------------------------------------------------
    # Table 3: Dynamic Green Times Allocation Comparison
    # ------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" 3. GREEN LIGHT TIMING ALLOCATION COMPARISON (seconds per approach)")
    print("-" * 95)
    print(f"{'Approach (Edge)':<18} | {'Initial Green (s)':<20} | {'PSO Green (s)':<20} | {'QAOA Green (s)':<20}")
    print("-" * 95)

    init_greens = _compute_green_times(graph, init_congestion, init_cycle_times, is_initial=True)

    sorted_approach_keys = sorted(list(init_greens.keys()))
    for e_id in sorted_approach_keys:
        i_g = init_greens.get(e_id, 0.0)
        p_g = pso_greens.get(e_id, i_g)
        q_g = qaoa_greens.get(e_id, 0.0)

        print(f"{e_id:<18} | {i_g:<20.1f} | {p_g:<20.1f} | {q_g:<20.1f}")

    print("-" * 95)

    # ------------------------------------------------------------------
    # Table 4: Overall Performance Metrics Comparison
    # ------------------------------------------------------------------
    print("\n" + "-" * 95)
    print(" 4. OVERALL SYSTEM PERFORMANCE METRICS")
    print("-" * 95)

    pk_init = max(init_occ.values()) * 100.0
    pk_pso = max(pso_occ.values()) * 100.0
    pk_qaoa = max(qaoa_occ.values()) * 100.0

    pk_red_pso = (pk_init - pk_pso) / pk_init * 100.0
    pk_red_qaoa = (pk_init - pk_qaoa) / pk_init * 100.0

    std_init = float(np.std(list(init_occ.values()))) * 100.0
    std_pso = float(np.std(list(pso_occ.values()))) * 100.0
    std_qaoa = float(np.std(list(qaoa_occ.values()))) * 100.0

    imp_pso = (std_init - std_pso) / std_init * 100.0 if std_init > 0 else 0.0
    imp_qaoa = (std_init - std_qaoa) / std_init * 100.0 if std_init > 0 else 0.0

    print(f"{'Metric':<35} | {'Initial State':<16} | {'PSO (pso.py)':<20} | {'QAOA (final_qaoa)':<22}")
    print("-" * 95)
    print(f"{'Peak Occupancy (%)':<35} | {pk_init:<16.1f}% | {pk_pso:<20.1f}% | {pk_qaoa:<22.1f}%")
    print(f"{'Peak Reduction (%)':<35} | {'N/A':<16} | {pk_red_pso:<20.1f}% | {pk_red_qaoa:<22.1f}%")
    print(f"{'Load Variance / Std Dev (%)':<35} | {std_init:<16.2f}% | {std_pso:<20.2f}% | {std_qaoa:<22.2f}%")
    print(f"{'Load Distribution Improvement (%)':<35} | {'N/A':<16} | {imp_pso:<20.1f}% | {imp_qaoa:<22.1f}%")
    print(f"{'Execution Latency (ms)':<35} | {'N/A':<16} | {pso_latency:<20.2f} ms | {qaoa_latency:<22.2f} ms")
    print("-" * 95)

    print("\n" + "=" * 95)
    print(" FRONTEND PSO COMPARISON COMPLETE")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    # 4-node network
    run_comparison(network_type="aditya")

    # 5-node network
    run_comparison(network_type="vedant")
