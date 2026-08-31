import os
import json
import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Directory Setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(SCRIPT_DIR, "mumbai_stqgcn_hierarchical_24k.npz")
JSON_PATH = os.path.join(SCRIPT_DIR, "hierarchical_graph_topology.json")
NODES_CSV_PATH = os.path.join(SCRIPT_DIR, "mumbai_hierarchical_nodes.csv")
EDGES_CSV_PATH = os.path.join(SCRIPT_DIR, "mumbai_hierarchical_edges.csv")

print("=================================================================")
print("GENERATING MUMBAI 24-NODE HIERARCHICAL DATASET (TREE + MESH + LINEAR)")
print("=================================================================\n")

# --- 1. Graph Topology Construction (24 Nodes, 3 Zones of 8 Nodes) ---
NUM_NODES = 24
NUM_TIMESTEPS = 10000

# Zone 1 (Tree Topology, 8 Nodes: 0..7)
# 0 (Root) -> 1, 2; 1 -> 3, 4; 2 -> 5, 6; 6 -> 7
TREE_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (1, 4),
    (2, 5), (2, 6),
    (6, 7)
] # 7 edges

# Zone 2 (Mesh/Grid Topology, 8 Nodes: 8..15)
# 2x4 grid layout with cross-mesh loops
MESH_EDGES = [
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (8, 12), (9, 13), (10, 14), (11, 15),
    (13, 9), (14, 10), (12, 9), (13, 10)
] # 14 edges

# Zone 3 (Linear Topology, 8 Nodes: 16..23)
# Sequential corridor pipeline
LINEAR_EDGES = [
    (16, 17), (17, 18), (18, 19), (19, 20),
    (20, 21), (21, 22), (22, 23)
] # 7 edges

# Inter-Zone Linking Edges
INTER_ZONE_EDGES = [
    (7, 8),   # Zone 1 Tree exit -> Zone 2 Mesh entry
    (15, 16)  # Zone 2 Mesh exit -> Zone 3 Linear entry
] # 2 edges

ALL_EDGES = TREE_EDGES + MESH_EDGES + LINEAR_EDGES + INTER_ZONE_EDGES
NUM_EDGES = len(ALL_EDGES) # 30 directed edges

print(f"Topology Summary: {NUM_NODES} Nodes (3 x 8), {NUM_EDGES} Directed Edges")
print(f"  - Zone 1 (Tree): 8 Nodes (V1-V8), {len(TREE_EDGES)} Intra-Zone Edges")
print(f"  - Zone 2 (Mesh): 8 Nodes (V9-V16), {len(MESH_EDGES)} Intra-Zone Edges")
print(f"  - Zone 3 (Linear): 8 Nodes (V17-V24), {len(LINEAR_EDGES)} Intra-Zone Edges")
print(f"  - Inter-Zone Bridges: {len(INTER_ZONE_EDGES)} Connecting Edges\n")

# Node Names & Types
node_list = []
zone_mapping = {
    "Zone_1_Tree_North": list(range(0, 8)),
    "Zone_2_Mesh_Central": list(range(8, 16)),
    "Zone_3_Linear_South": list(range(16, 24))
}

for i in range(24):
    if i < 8:
        z_name = "Zone1_Tree"
        n_type = 0 if i == 0 else 1
    elif i < 16:
        z_name = "Zone2_Mesh"
        n_type = 2 # Complex interchange
    else:
        z_name = "Zone3_Linear"
        n_type = 1 # Corridor segment
        
    node_list.append({
        "id": i,
        "name": f"V{i+1}_{z_name}_Node{i%8 + 1}",
        "zone": z_name,
        "type": n_type
    })

# --- 2. Temporal Features (10,000 5-Minute Timesteps) ---
time_idx = np.arange(NUM_TIMESTEPS)
sin_diurnal = np.sin(2 * np.pi * time_idx / 288) # 288 5-min intervals per day
cos_diurnal = np.cos(2 * np.pi * time_idx / 288)
day_of_week = (time_idx // 288) % 7
is_weekend = np.where(day_of_week >= 5, 1.0, 0.0)

T_temp = np.stack([sin_diurnal, cos_diurnal, day_of_week / 6.0, is_weekend], axis=1) # (10000, 4)

# --- 3. Generate Physics-Informed Node Features (10000, 24, 6) ---
# Features: [Volume/Flow, Speed, Density, Weather_Rain, Incident_Flag, Signal_Timing]
X_node = np.zeros((NUM_TIMESTEPS, NUM_NODES, 6), dtype=np.float32)

for i in range(NUM_NODES):
    peak_morning = np.exp(-((time_idx % 288 - 96) ** 2) / 200.0) * 1200
    peak_evening = np.exp(-((time_idx % 288 - 216) ** 2) / 250.0) * 1500
    base_flow = 800 + 400 * sin_diurnal + peak_morning + peak_evening + np.random.normal(0, 50, NUM_TIMESTEPS)
    
    if i < 8: # Tree topology (Suburban inflow)
        flow = base_flow * (1.2 - 0.05 * (i % 8))
    elif i < 16: # Mesh topology (Central loop congestion)
        flow = base_flow * (1.4 + 0.1 * np.sin(i))
    else: # Linear topology (South island bottleneck)
        flow = base_flow * (1.1 + 0.03 * (i % 8))
        
    flow = np.clip(flow, 100, 3000)
    
    free_speed = 65.0 - 0.5 * (i % 5)
    speed = np.clip(free_speed * (1.0 - (flow / 3200.0) ** 1.8) + np.random.normal(0, 2, NUM_TIMESTEPS), 5.0, 80.0)
    density = np.clip(flow / (speed + 1e-3), 5.0, 150.0)
    monsoon_wave = np.clip(np.sin(2 * np.pi * time_idx / 2000.0) * 15.0 + np.random.exponential(2, NUM_TIMESTEPS), 0, 50)
    incidents = np.random.choice([0.0, 1.0], size=NUM_TIMESTEPS, p=[0.97, 0.03])
    signal = 0.5 + 0.2 * np.cos(2 * np.pi * time_idx / 12)
    
    X_node[:, i, 0] = flow
    X_node[:, i, 1] = speed
    X_node[:, i, 2] = density
    X_node[:, i, 3] = monsoon_wave
    X_node[:, i, 4] = incidents
    X_node[:, i, 5] = signal

# --- 4. Generate Physics-Informed Edge Features (10000, 30, 6) & Target Flow ---
# Features: [Capacity, Length, PCI, Speed_Limit, Lanes, Congestion_Ratio]
E_edge = np.zeros((NUM_TIMESTEPS, NUM_EDGES, 6), dtype=np.float32)
Y_target_flow = np.zeros((NUM_TIMESTEPS, NUM_EDGES), dtype=np.float32)
Y_target_speed = np.zeros((NUM_TIMESTEPS, NUM_EDGES), dtype=np.float32)

edge_metadata = []

for e_idx, (src, dst) in enumerate(ALL_EDGES):
    length_km = 1.5 + 0.3 * (e_idx % 4)
    lanes = 4 if (src in range(8, 16)) else 3 # Mesh has more lanes
    capacity = lanes * 900.0
    pci = 75.0 + 5.0 * (e_idx % 3)
    speed_limit = 60.0 if lanes == 3 else 70.0
    
    edge_flow = 0.55 * X_node[:, src, 0] + 0.45 * X_node[:, dst, 0] + np.random.normal(0, 30, NUM_TIMESTEPS)
    edge_flow = np.clip(edge_flow, 100, capacity * 1.25)
    edge_speed = 0.5 * X_node[:, src, 1] + 0.5 * X_node[:, dst, 1]
    congestion_ratio = edge_flow / capacity
    
    E_edge[:, e_idx, 0] = capacity
    E_edge[:, e_idx, 1] = length_km
    E_edge[:, e_idx, 2] = pci
    E_edge[:, e_idx, 3] = speed_limit
    E_edge[:, e_idx, 4] = lanes
    E_edge[:, e_idx, 5] = congestion_ratio
    
    Y_target_flow[:, e_idx] = edge_flow
    Y_target_speed[:, e_idx] = edge_speed
    
    edge_metadata.append({
        "id": e_idx,
        "src": src,
        "dst": dst,
        "name": f"E{e_idx}_V{src+1}->V{dst+1}",
        "capacity": capacity,
        "length_km": length_km,
        "lanes": lanes
    })

# --- 5. Save Outputs ---
print(f"Saving NPZ Archive to: {NPZ_PATH}")
np.savez_compressed(
    NPZ_PATH,
    node_features=X_node,
    edge_features=E_edge,
    temporal_features=T_temp,
    target_flow=Y_target_flow,
    target_speed=Y_target_speed
)

json_data = {
    "num_nodes": NUM_NODES,
    "num_edges": NUM_EDGES,
    "num_timesteps": NUM_TIMESTEPS,
    "nodes": node_list,
    "edges": edge_metadata,
    "zones": zone_mapping
}
with open(JSON_PATH, "w") as f:
    json.dump(json_data, f, indent=4)
print(f"Saved JSON Topology to: {JSON_PATH}")

pd.DataFrame(node_list).to_csv(NODES_CSV_PATH, index=False)
pd.DataFrame(edge_metadata).to_csv(EDGES_CSV_PATH, index=False)
print(f"Saved CSV summaries to: {NODES_CSV_PATH} and {EDGES_CSV_PATH}")

print("\n=================================================================")
print("HIERARCHICAL 24-NODE DATASET GENERATION COMPLETE!")
print("=================================================================")
