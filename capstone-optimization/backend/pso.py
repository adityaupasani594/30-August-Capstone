"""
pso.py
======
Discrete / Binary Particle Swarm Optimization (BPSO) engine for traffic rerouting.

Operates on the EXACT SAME topology analysis, packet definitions, candidate routes,
route-decoding logic (`apply_solution`), and QUBO objective function as QAOA (`final_qaoa_with_signal.py`).

Algorithm Overview:
    Each particle encodes candidate packet rerouting bitstrings for all N binary variables:
        position = [x_1, x_2, ..., x_N] ∈ R^N
        bitstring = [b_1, b_2, ..., b_N] ∈ {0, 1}^N via sigmoid thresholding

    Velocities & Positions:
        v_i = w * v_i + c1 * r1 * (pbest_i - position_i) + c2 * r2 * (gbest_i - position_i)
        position_i = clip(position_i + v_i, -10.0, 10.0)

    Route Decoding & Objective Evaluation:
        Evaluates the exact QUBO cost H_B(b) and post-optimization edge loads via `apply_solution`.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional

from graph import TrafficGraph
from predictions import (
    INITIAL_CONGESTION,
    INITIAL_CYCLE_TIMES,
    ADITYA_INITIAL_CYCLE_TIMES,
)
from final_qaoa_with_signal import (
    build_network,
    detect_congestion,
    generate_alternate_routes,
    generate_packets,
    build_qubo,
    apply_solution,
    evaluate_network,
    optimize_signals,
    clean_edge_key,
    normalize_edge,
)


class Particle:
    """A single particle in the Binary PSO swarm representing route decisions."""

    def __init__(self, n_dims: int, rng: np.random.Generator) -> None:
        self.position: np.ndarray = rng.uniform(-2.0, 2.0, n_dims)
        self.velocity: np.ndarray = rng.uniform(-1.0, 1.0, n_dims)
        self.best_position: np.ndarray = self.position.copy()
        self.best_bitstring: Tuple[int, ...] = self.to_bitstring()
        self.best_fitness: float = float("inf")

    def to_bitstring(self) -> Tuple[int, ...]:
        sig = 1.0 / (1.0 + np.exp(-np.clip(self.position, -10.0, 10.0)))
        return tuple(int(b) for b in (sig >= 0.5))

    def update_velocity(
        self, global_best_position: np.ndarray, w: float, c1: float, c2: float, rng: np.random.Generator
    ) -> None:
        n = len(self.position)
        r1 = rng.uniform(0.0, 1.0, n)
        r2 = rng.uniform(0.0, 1.0, n)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
        self.velocity = np.clip(self.velocity, -6.0, 6.0)

    def update_position(self) -> None:
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, -10.0, 10.0)


class PSO:
    def __init__(
        self,
        n_particles: int = 30,
        max_iter: int = 20,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        threshold_factor: float = 0.6,
        seed: int = 42,
    ) -> None:
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.threshold_factor = threshold_factor
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        graph: TrafficGraph,
        initial_congestion: Dict[str, float] | None = None,
        initial_cycle_times: Dict[str, float] | None = None,
    ) -> dict:

        if initial_congestion is None:
            initial_congestion = graph.get_initial_predictions()
        if initial_cycle_times is None:
            initial_cycle_times = ADITYA_INITIAL_CYCLE_TIMES.copy() if graph.network_type == "aditya" else INITIAL_CYCLE_TIMES.copy()

        user_capacities = {e.id: e.capacity for e in graph.edges}

        # 1. Build network & normalize edge representations
        G, capacities, predicted_loads = build_network(
            graph=None,
            capacities=user_capacities,
            predicted_loads=initial_congestion,
        )

        # 2. Detect congestion
        occupancy, congested, underutilized = detect_congestion(capacities, predicted_loads)

        # 3. Generate alternate detour routes (top 2 least occupied)
        alt_routes = generate_alternate_routes(G, congested, occupancy, max_cutoff=3, top_k=2)

        # 4. Generate packets
        packets, variables = generate_packets(
            congested, predicted_loads, capacities, alt_routes,
            target=0.50, packet_size=150.0, max_packets=10
        )

        # Build lookup from tuple edge keys to graph string edge IDs
        edge_map = {}
        for edge in graph.edges:
            norm = normalize_edge(edge.source, edge.target)
            edge_map[norm] = edge.id

        N = len(variables)
        non_ref_edges = graph.get_non_reference_edges()
        desired_congestion = (
            sum(e.threshold for e in non_ref_edges) / len(non_ref_edges) if non_ref_edges else 0.0
        )

        # Fallback if no congested packets generated
        if N == 0 or not variables:
            opt_cong = {e.id: float(initial_congestion.get(e.id, 0.0)) for e in graph.edges}
            opt_ct = initial_cycle_times.copy()
            return {
                "optimized_cycle_times": opt_ct,
                "optimized_congestion":  opt_cong,
                "fitness_history":       [0.0],
                "initial_fitness":       0.0,
                "final_fitness":         0.0,
                "iterations":            0,
                "desired_congestion":    desired_congestion,
                "gbest_signal_plan":     [],
                "gbest_bitstring":       (),
            }

        # 5. Build QUBO
        qubo_problem = build_qubo(variables, capacities, occupancy, target=0.50, packet_size=150.0)
        C = qubo_problem["constant"]
        L = qubo_problem["linear"]
        Q = qubo_problem["quadratic"]
        var_names = [v["name"] for v in variables]

        def compute_qubo_cost(bitstring: Tuple[int, ...]) -> float:
            val = C
            for i, name in enumerate(var_names):
                val += L.get(name, 0.0) * bitstring[i]
            for i in range(N):
                for j in range(i + 1, N):
                    pair = tuple(sorted((var_names[i], var_names[j])))
                    val += Q.get(pair, 0.0) * bitstring[i] * bitstring[j]
            return float(val)

        def evaluate_bitstring(bitstring: Tuple[int, ...]) -> Tuple[float, Dict[str, float], Dict[str, float]]:
            cost = compute_qubo_cost(bitstring)
            up_loads = apply_solution(packets, bitstring, predicted_loads, packet_size=150.0)

            # Map updated tuple flows back to string edge IDs in graph
            opt_cong = {}
            for edge in graph.edges:
                norm_e = normalize_edge(edge.source, edge.target)
                opt_cong[edge.id] = float(up_loads.get(norm_e, initial_congestion.get(edge.id, 0.0)))

            # Retune each junction cycle from its post-optimization incoming occupancy.
            opt_ct = {}
            for node_id in graph.nodes:
                incoming = graph.get_incoming_edges(node_id)
                if incoming:
                    occupancy = sum(opt_cong.get(edge.id, 0.0) / max(edge.capacity, 1.0) for edge in incoming) / len(incoming)
                else:
                    occupancy = 0.5
                baseline = float(initial_cycle_times.get(node_id, 60.0))
                opt_ct[node_id] = round(float(np.clip(baseline + (occupancy - 0.50) * 30.0, 30.0, 120.0)), 2)

            return cost, opt_ct, opt_cong

        # Initial baseline fitness (bitstring = all 0s)
        initial_bitstring = tuple([0] * N)
        initial_fitness, _, _ = evaluate_bitstring(initial_bitstring)

        # Initialise Swarm
        particles: List[Particle] = [Particle(N, self.rng) for _ in range(self.n_particles)]

        global_best_fitness: float = float("inf")
        global_best_position: np.ndarray = particles[0].position.copy()
        global_best_bitstring: Tuple[int, ...] = initial_bitstring

        # Initial evaluation
        for p in particles:
            bits = p.to_bitstring()
            fit, _, _ = evaluate_bitstring(bits)
            p.best_fitness = fit
            p.best_position = p.position.copy()
            p.best_bitstring = bits
            if fit < global_best_fitness:
                global_best_fitness = fit
                global_best_position = p.position.copy()
                global_best_bitstring = bits

        fitness_history: List[float] = [initial_fitness]

        # PSO Main Loop
        for _iter in range(self.max_iter):
            for p in particles:
                p.update_velocity(global_best_position, self.w, self.c1, self.c2, self.rng)
                p.update_position()

                bits = p.to_bitstring()
                fit, _, _ = evaluate_bitstring(bits)

                if fit < p.best_fitness:
                    p.best_fitness = fit
                    p.best_position = p.position.copy()
                    p.best_bitstring = bits

                if fit < global_best_fitness:
                    global_best_fitness = fit
                    global_best_position = p.position.copy()
                    global_best_bitstring = bits

            fitness_history.append(global_best_fitness)

        # Extract Best Results
        final_fitness, optimized_cycle_times, optimized_congestion = evaluate_bitstring(global_best_bitstring)

        return {
            "optimized_cycle_times": optimized_cycle_times,
            "optimized_congestion":  optimized_congestion,
            "fitness_history":       fitness_history,
            "initial_fitness":       initial_fitness,
            "final_fitness":         final_fitness,
            "iterations":            self.max_iter,
            "desired_congestion":    desired_congestion,
            "gbest_signal_plan":     list(global_best_bitstring),
            "gbest_bitstring":       global_best_bitstring,
        }
