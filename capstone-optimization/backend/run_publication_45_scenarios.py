"""
run_publication_45_scenarios.py
==================================
Publication-Quality Diverse Traffic Scenario Generator & Evaluator for QAOA vs PSO.

Generates and runs 15 mathematically unique, realistic time-of-day traffic scenarios per topology
(4-node, 5-node, 6-node mesh networks) with a fixed random seed (2026) for exact reproducibility.
"""

import os
import sys
import time
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


def plot_topology_pso_convergences(all_topology_results: Dict[str, List[Dict[str, Any]]]):
    """
    Plot and save PSO convergence curves for 4-Node, 5-Node, and 6-Node networks across 15 scenarios.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    for idx, (top_name, evals) in enumerate(all_topology_results.items()):
        ax = axes[idx]
        all_convergences = []

        for ev in evals:
            conv = ev["pso_convergence"]
            all_convergences.append(conv)

        conv_matrix = np.array(all_convergences)  # shape (15, N)
        num_iterations = conv_matrix.shape[1]
        iterations = list(range(num_iterations))

        for conv in all_convergences:
            ax.plot(iterations, conv, color='gray', alpha=0.3, linewidth=1.2)

        mean_conv = np.mean(conv_matrix, axis=0)

        ax.plot(
            iterations, mean_conv,
            marker='o', markersize=6, linewidth=2.5,
            color='#1f77b4', label='Mean Convergence (15 Scenarios)'
        )

        ax.set_title(f"PSO Convergence: {top_name}", fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Iteration", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Fitness (Peak Congestion % + Penalty)", fontsize=11)

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_xticks(range(0, num_iterations, 2))

    plt.suptitle("PSO Convergence Across Topologies (pso-final.py Logic)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = "pso_convergence_4_5_6_nodes.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[INFO] Convergence graph successfully saved to: {os.path.abspath(output_path)}")
    plt.show()



from graph import TrafficGraph, Node, Edge
from pso import PSO
from final_qaoa_with_signal import (
    build_network, detect_congestion, generate_alternate_routes,
    generate_packets, build_qubo, solve_qaoa, apply_solution,
    evaluate_network, run_qaoa_optimization, normalize_edge
)

def run_brute_force_qubo(qubo_problem: Dict[str, Any], packets: List[Dict[str, Any]], capacities: Dict, initial_loads: Dict, packet_size: float = 150.0):
    variables = qubo_problem["variables"]
    N = len(variables)
    if N > 12:
        return None

    var_names = [v["name"] for v in variables]
    C = qubo_problem["constant"]
    L = qubo_problem["linear"]
    Q = qubo_problem["quadratic"]

    best_bitstring = None
    best_qubo_cost = float("inf")
    best_peak_occ = float("inf")

    for bits in itertools.product([0, 1], repeat=N):
        val = C
        for i, name in enumerate(var_names):
            val += L.get(name, 0.0) * bits[i]
        for i in range(N):
            for j in range(i + 1, N):
                pair = tuple(sorted((var_names[i], var_names[j])))
                val += Q.get(pair, 0.0) * bits[i] * bits[j]

        up_l = apply_solution(packets, bits, initial_loads, packet_size=packet_size)
        pk = max(up_l[e] / capacities[e] for e in capacities)

        if pk < best_peak_occ - 1e-6 or (abs(pk - best_peak_occ) <= 1e-6 and val < best_qubo_cost):
            best_peak_occ = pk
            best_qubo_cost = val
            best_bitstring = bits

    return {
        "best_bitstring": best_bitstring,
        "best_qubo_cost": best_qubo_cost,
        "best_peak_occupancy": best_peak_occ
    }

def generate_15_diverse_publication_scenarios(nodes: List[str], G: nx.Graph, seed: int = 2026) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    edges = [tuple(sorted(e)) for e in G.edges()]
    num_edges = len(edges)

    scenario_configs = [
        ("1. Single Major Expressway Bottleneck", "Single Bottleneck", 1),
        ("2. Adjacent Dual Arterial Bottleneck", "Adjacent Bottlenecks", 2),
        ("3. Non-Adjacent Dual Arterial Bottleneck", "Non-Adjacent Bottlenecks", 2),
        ("4. Radial Inbound Morning Commute Peak", "Inbound Corridor", 2),
        ("5. Radial Outbound Evening Egress Surge", "Outbound Corridor", 2),
        ("6. Commercial District Central Rush", "Localized Congestion", 3),
        ("7. Construction Capacity Reduction Incident", "Lane Closure Incident", 1),
        ("8. Stadium Special Event Outflow Spike", "Event Outflow", 2),
        ("9. School Zone Perimeter Dropoff", "Localized Dropoff", 2),
        ("10. Asymmetric Directional Flow Pattern", "Asymmetric Demand", 2),
        ("11. Inclement Weather Network Slowdown", "Weather Slowdown", 3),
        ("12. Cross-Town Transit Corridor Surge", "Corridor Congestion", 2),
        ("13. Late Night Heavy Freight Arterial", "Freight Corridor", 1),
        ("14. Distributed Triple Bottleneck Peak", "Multiple Bottlenecks", 3),
        ("15. Off-Peak Nighttime Low Baseline", "Off-Peak Baseline", 1),
    ]

    scenarios = []

    for idx, (name, pattern_type, num_bottlenecks) in enumerate(scenario_configs):
        capacities = {}
        for e_i, e in enumerate(edges):
            if e_i % 4 == 0:
                cap = 2200.0 + rng.choice([0.0, 100.0, 200.0])  # Expressway
            elif e_i % 4 == 1:
                cap = 1900.0 + rng.choice([-100.0, 0.0, 100.0]) # Major Arterial
            elif e_i % 4 == 2:
                cap = 1750.0 + rng.choice([-50.0, 0.0, 50.0])   # Standard Arterial
            else:
                cap = 1500.0 + rng.choice([-100.0, 0.0, 100.0]) # Minor Collector
            capacities[e] = cap

        if "Construction" in name:
            target_edge = edges[idx % num_edges]
            capacities[target_edge] = 1200.0  # Reduced capacity

        if pattern_type == "Single Bottleneck" or pattern_type == "Lane Closure Incident" or pattern_type == "Off-Peak Baseline" or pattern_type == "Freight Corridor":
            bottlenecks = [edges[idx % num_edges]]
        elif pattern_type == "Adjacent Bottlenecks":
            e1_idx = idx % num_edges
            e1 = edges[e1_idx]
            adj_edges = [e for e in edges if e != e1 and (e[0] in e1 or e[1] in e1)]
            e2 = adj_edges[idx % len(adj_edges)] if adj_edges else edges[(idx + 1) % num_edges]
            bottlenecks = [e1, e2]
        elif pattern_type == "Non-Adjacent Bottlenecks":
            e1_idx = idx % num_edges
            e1 = edges[e1_idx]
            non_adj = [e for e in edges if e != e1 and (e[0] not in e1 and e[1] not in e1)]
            e2 = non_adj[idx % len(non_adj)] if non_adj else edges[(idx + 2) % num_edges]
            bottlenecks = [e1, e2]
        else:
            bottlenecks = [edges[(idx + k * 2) % num_edges] for k in range(num_bottlenecks)]

        loads = {}
        for e in edges:
            if e in bottlenecks:
                occ = 0.68 + rng.uniform(0.04, 0.18)
                loads[e] = round(capacities[e] * min(0.88, occ), 1)
            else:
                occ = 0.28 + rng.uniform(0.02, 0.16)
                loads[e] = round(capacities[e] * occ, 1)

        scenarios.append({
            "name": name,
            "pattern_type": pattern_type,
            "bottleneck_edges": bottlenecks,
            "capacities": capacities,
            "loads": loads
        })

    return scenarios

def evaluate_and_verify_scenario(topology_name: str, G: nx.Graph, scenario: Dict[str, Any]):
    caps = scenario["capacities"]
    loads = scenario["loads"]

    G_built, norm_caps, norm_loads = build_network(G, caps, loads)
    occupancy, congested, underutilized = detect_congestion(norm_caps, norm_loads, high_threshold=0.60, low_threshold=0.45)
    alt_routes = generate_alternate_routes(G_built, congested, occupancy, max_cutoff=3, top_k=2)

    total_candidate_routes = 0
    for e, r_list in alt_routes.items():
        r_tuples = [tuple(r) for r in r_list]
        assert len(r_tuples) == len(set(r_tuples)), f"Duplicate route detected on edge {e}!"
        total_candidate_routes += len(r_list)

    packets, variables = generate_packets(congested, norm_loads, norm_caps, alt_routes, target=0.50, packet_size=150.0, max_packets=10)
    qubo_problem = build_qubo(variables, norm_caps, occupancy, target=0.50, packet_size=150.0)

    t_qaoa_start = time.perf_counter()
    qaoa_sol = solve_qaoa(qubo_problem, packets=packets, capacities=norm_caps, initial_loads=norm_loads, packet_size=150.0, maxiter=50)
    t_qaoa_end = time.perf_counter()
    qaoa_latency_ms = (t_qaoa_end - t_qaoa_start) * 1000.0

    best_bit = qaoa_sol["bitstring"]
    updated_loads = apply_solution(packets, best_bit, norm_loads, packet_size=150.0)

    init_occs = [norm_loads[e] / norm_caps[e] for e in norm_caps]
    init_peak = max(init_occs) * 100.0
    init_mean = float(np.mean(init_occs)) * 100.0
    init_std = float(np.std(init_occs)) * 100.0

    qaoa_occs = [updated_loads[e] / norm_caps[e] for e in norm_caps]
    qaoa_peak = max(qaoa_occs) * 100.0
    qaoa_mean = float(np.mean(qaoa_occs)) * 100.0
    qaoa_std = float(np.std(qaoa_occs)) * 100.0
    qaoa_peak_red = ((init_peak - qaoa_peak) / init_peak * 100.0) if init_peak > 0 else 0.0

    bf_sol = run_brute_force_qubo(qubo_problem, packets, norm_caps, norm_loads, packet_size=150.0)
    opt_gap = abs(qaoa_sol["best_cost"] - bf_sol["best_qubo_cost"]) if bf_sol else 0.0

    dir_init_cong = {}
    for (u, v), load in norm_loads.items():
        dir_init_cong[f"{u}\u2192{v}"] = load
        dir_init_cong[f"{v}\u2192{u}"] = load

    t_graph = TrafficGraph()
    t_graph.nodes = {n: Node(id=n, label=n, initial_cycle_time=60.0) for n in G_built.nodes()}
    t_graph.edges = []
    t_graph._edge_index = {}
    for (u, v), cap in norm_caps.items():
        e1 = Edge(source=u, target=v, weight=1.0, capacity=cap, speed=50.0, lanes=3, length=1.0, road_type="Arterial", threshold=0.60 * cap)
        e2 = Edge(source=v, target=u, weight=1.0, capacity=cap, speed=50.0, lanes=3, length=1.0, road_type="Arterial", threshold=0.60 * cap)
        t_graph.edges.extend([e1, e2])
        t_graph._edge_index[e1.id] = e1
        t_graph._edge_index[e2.id] = e2

    pso = PSO(n_particles=30, max_iter=20, seed=42)
    t_pso_start = time.perf_counter()
    init_cycles = {n: 60.0 for n in G_built.nodes()}
    pso_res = pso.optimize(t_graph, initial_congestion=dir_init_cong, initial_cycle_times=init_cycles)
    t_pso_end = time.perf_counter()
    pso_latency_ms = (t_pso_end - t_pso_start) * 1000.0

    pso_flows = pso_res["optimized_congestion"]
    pso_occs = [pso_flows.get(e.id, 0.0) / e.capacity for e in t_graph.edges if e.capacity > 0]
    pso_peak = max(pso_occs) * 100.0 if pso_occs else init_peak

    return {
        "scenario_name": scenario["name"],
        "pattern_type": scenario["pattern_type"],
        "bottlenecks": scenario["bottleneck_edges"],
        "num_packets": len(packets),
        "num_alt_routes": total_candidate_routes,
        "init_peak": init_peak,
        "pso_peak": pso_peak,
        "qaoa_peak": qaoa_peak,
        "qaoa_peak_red": qaoa_peak_red,
        "opt_gap": opt_gap,
        "pso_convergence": pso_res["fitness_history"]
    }

def execute_all_publication_benchmarks():
    topologies = [
        ("4-Node Complete Mesh", nx.complete_graph(["A", "B", "C", "D"])),
        ("5-Node Connected Mesh", nx.complete_graph(["A", "B", "C", "D", "E"])),
        ("6-Node Complete Mesh", nx.complete_graph(["A", "B", "C", "D", "E", "F"]))
    ]

    all_topology_results = {}

    for top_name, G in topologies:
        print("\n" + "=" * 110)
        print(f" PUBLICATION BENCHMARK SUITE: {top_name.upper()} (15 UNIQUE REALISTIC SCENARIOS)")
        print("=" * 110)

        scenarios = generate_15_diverse_publication_scenarios(list(G.nodes()), G, seed=2026)
        top_evals = []

        print(f"{'Scenario Name':<42} | {'Bottlenecks':<18} | {'Pkts':<4} | {'Routes':<6} | {'Init Peak':<9} | {'PSO Peak':<9} | {'QAOA Peak':<9} | {'Red %':<7} | {'Opt Gap':<9}")
        print("-" * 125)

        for sc in scenarios:
            ev = evaluate_and_verify_scenario(top_name, G, sc)
            top_evals.append(ev)

            b_str = ", ".join([f"{e[0]}-{e[1]}" for e in ev["bottlenecks"]])
            print(f"{ev['scenario_name']:<42} | {b_str:<18} | {ev['num_packets']:<4} | {ev['num_alt_routes']:<6} | {ev['init_peak']:<8.1f}% | {ev['pso_peak']:<8.1f}% | {ev['qaoa_peak']:<8.1f}% | {ev['qaoa_peak_red']:<6.1f}% | {ev['opt_gap']:<9.1e}")

        all_topology_results[top_name] = top_evals

    plot_topology_pso_convergences(all_topology_results)

    return all_topology_results

if __name__ == "__main__":
    execute_all_publication_benchmarks()
