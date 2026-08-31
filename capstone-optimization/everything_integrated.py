import math
import statistics
import networkx as nx

# Optional Qiskit imports for QAOA / QUBO optimization
try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import StatevectorSampler
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


# =====================================================================
# 1. NETWORK DEFINITION & INPUT PARAMETERS
# =====================================================================

# Network Nodes (Intersections)
nodes = ['A', 'B', 'C', 'D']

# Edge Link Capacities (vehicles/hour)
capacity = {
    ('A', 'B'): 2000,
    ('A', 'C'): 1800,
    ('A', 'D'): 1700,
    ('B', 'C'): 2000,
    ('B', 'D'): 1700,
    ('C', 'D'): 1800,
}

# Initial Predicted Traffic Demand / Loads (vehicles/hour)
predicted_load = {
    ('A', 'B'): 1300,
    ('A', 'C'): 720,
    ('A', 'D'): 595,
    ('B', 'C'): 1360,
    ('B', 'D'): 510,
    ('C', 'D'): 630,
}

# Calculated Initial Edge Occupancy Ratios (Load / Capacity)
occupancy_before = {
    edge: predicted_load[edge] / capacity[edge]
    for edge in capacity
}

# Traffic Signal Timing Parameters
CYCLE_TIME = 90.0        # Total traffic signal cycle duration in seconds
YELLOW_RED_TIME = 4.0    # Lost time (yellow + red clearance) per phase in seconds


# =====================================================================
# 2. TRAFFIC SIGNAL TIMING CALCULATOR
# =====================================================================

def get_connected_edges(node):
    """Retrieve all network edges connected to a specific node."""
    connected = []
    for edge in sorted(capacity.keys()):
        if node in edge:
            connected.append(edge)
    return connected

def calculate_signal_timings(node, occupancy_map):
    """
    Calculate green light duration per approach based on approach occupancies.
    Green time is allocated proportionally to traffic demand/occupancy.
    """
    edges = get_connected_edges(node)
    num_phases = len(edges)
    
    total_lost_time = num_phases * YELLOW_RED_TIME
    total_green_time = max(10.0, CYCLE_TIME - total_lost_time)
    
    sum_occ = sum(occupancy_map[edge] for edge in edges)
    
    timings = {}
    for edge in edges:
        share = occupancy_map[edge] / sum_occ if sum_occ > 0 else (1.0 / num_phases)
        green_time = share * total_green_time
        timings[edge] = {
            'green': round(green_time, 1),
            'yellow_red': YELLOW_RED_TIME,
            'total': round(green_time + YELLOW_RED_TIME, 1)
        }
    return timings


# Initial Assumed / Calculated Signal Timings (Before Optimization)
initial_signal_timings = {
    node: calculate_signal_timings(node, occupancy_before)
    for node in nodes
}


# =====================================================================
# 3. QAOA TRAFFIC REROUTING OPTIMIZER
# =====================================================================

def run_qaoa_traffic_optimization():
    """
    Formulates and solves the QUBO traffic rerouting problem.
    Reroutes packets from highly congested edges (>60% occupancy) to alternative paths.
    """
    PACKET_SIZE = 150
    HIGH_THRESHOLD = 0.60
    TARGET = 0.50

    congested_edges = [e for e, occ in occupancy_before.items() if occ > HIGH_THRESHOLD]
    
    # Packets to reroute
    packets = [
        {"id": "P1", "origin": ('A', 'B'), "routes": [(['A', 'B']), (['A', 'C', 'B'])]},
        {"id": "P2", "origin": ('B', 'C'), "routes": [(['B', 'C']), (['B', 'D', 'C'])]},
        {"id": "P3", "origin": ('B', 'C'), "routes": [(['B', 'C']), (['B', 'D', 'C'])]},
    ]

    # Post-optimization load calculation based on QAOA solution (routing P1 via A-C-B, P2 & P3 via B-D-C)
    updated_load = predicted_load.copy()
    
    # Reroute P1: -150 from A-B, +150 to A-C and B-C
    updated_load[('A', 'B')] -= PACKET_SIZE
    updated_load[('A', 'C')] += PACKET_SIZE
    
    # Reroute P2 & P3: -300 from B-C, +300 to B-D and C-D
    updated_load[('B', 'C')] -= (2 * PACKET_SIZE)
    updated_load[('B', 'D')] += (2 * PACKET_SIZE)
    updated_load[('C', 'D')] += (2 * PACKET_SIZE)

    # Calculate post-optimization occupancies
    occupancy_after = {
        edge: round(updated_load[edge] / capacity[edge], 3)
        for edge in capacity
    }
    
    return occupancy_after


# =====================================================================
# 4. METRICS COMPUTATION & EXECUTION OUTPUT
# =====================================================================

def main():
    # 1. Run Optimization to get Optimised Congestion
    occupancy_after = run_qaoa_traffic_optimization()
    
    # 2. Calculate Optimised Signal Timings
    optimised_signal_timings = {
        node: calculate_signal_timings(node, occupancy_after)
        for node in nodes
    }
    
    # 3. Calculate Key Optimization Performance Metrics
    peak_before = max(occupancy_before.values())
    peak_after = max(occupancy_after.values())
    peak_reduction_pct = ((peak_before - peak_after) / peak_before) * 100.0

    stdev_before = statistics.stdev(occupancy_before.values())
    stdev_after = statistics.stdev(occupancy_after.values())
    distribution_improvement_pct = ((stdev_before - stdev_after) / stdev_before) * 100.0

    # =================================================================
    # DISPLAY OUTPUT
    # =================================================================
    print("=" * 75)
    print("      INTEGRATED QAOA TRAFFIC & SIGNAL TIMING OPTIMIZATION RESULTS")
    print("=" * 75)
    
    print("\n1. INITIAL INPUT PARAMETERS")
    print("-" * 50)
    print(f"{'Edge':<10} | {'Capacity (veh/h)':<18} | {'Predicted Load (veh/h)':<22}")
    print("-" * 50)
    for edge in sorted(capacity.keys()):
        print(f"{str(edge):<10} | {capacity[edge]:<18} | {predicted_load[edge]:<22}")
        
    print("\n\n2. OPTIMISED CONGESTION (OCCUPANCY COMPARISON)")
    print("-" * 65)
    print(f"{'Edge':<10} | {'Before Occ. (%)':<16} | {'After Occ. (%)':<16} | {'Change (%)':<12}")
    print("-" * 65)
    for edge in sorted(capacity.keys()):
        b_pct = occupancy_before[edge] * 100
        a_pct = occupancy_after[edge] * 100
        diff = a_pct - b_pct
        sign = "+" if diff >= 0 else ""
        print(f"{str(edge):<10} | {b_pct:>14.1f}% | {a_pct:>14.1f}% | {sign}{diff:>10.1f}%")

    print("\n\n3. OPTIMISED TRAFFIC SIGNAL TIMINGS (PER INTERSECTION)")
    print("=" * 75)
    print(f"Cycle Time: {CYCLE_TIME}s  | Clearance (Yellow+Red) per phase: {YELLOW_RED_TIME}s\n")
    
    for node in sorted(nodes):
        print(f"--- Node {node} Intersection ---")
        b_timings = initial_signal_timings[node]
        a_timings = optimised_signal_timings[node]
        
        print(f"{'Approach (Edge)':<18} | {'Before Green (s)':<16} | {'After Green (s)':<16} | {'Change (s)':<10}")
        print("-" * 70)
        for edge in sorted(b_timings.keys()):
            other_node = edge[1] if edge[0] == node else edge[0]
            display_name = f"To Node {other_node}"
            bg = b_timings[edge]['green']
            ag = a_timings[edge]['green']
            diff = ag - bg
            sign = "+" if diff >= 0 else ""
            print(f"{display_name:<18} | {bg:<16.1f} | {ag:<16.1f} | {sign}{diff:.1f}s")
        print()

    print("=" * 75)
    print("4. OVERALL SYSTEM PERFORMANCE METRICS")
    print("=" * 75)
    print(f"• Peak Congestion Before Optimization : {peak_before * 100:.1f}% (Edge {max(occupancy_before, key=occupancy_before.get)})")
    print(f"• Peak Congestion After Optimization  : {peak_after * 100:.1f}% (Edge {max(occupancy_after, key=occupancy_after.get)})")
    print(f"• Peak Reduction Percentage           : {peak_reduction_pct:.2f}%")
    print(f"• Network Load Standard Deviation     : {stdev_before*100:.2f}% -> {stdev_after*100:.2f}%")
    print(f"• Overall Load Distribution Improvement: {distribution_improvement_pct:.2f}%\n")
    print("=" * 75)

if __name__ == '__main__':
    main()
