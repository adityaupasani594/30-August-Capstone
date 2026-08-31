import numpy as np
import pandas as pd
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

NUM_TIMESTEPS = 10000
TIME_INTERVAL_MINS = 5

# --- Graph Definition (8 Nodes, 11 Directed Edges) ---
# Nodes:
# V1: Dahisar/Borivali Toll (North Entrance)
# V2: WEH-JVLR Interchange (Andheri East)
# V3: Dadar TT Circle
# V4: Western Express Highway - Jogeshwari
# V5: BKC Connector / Kalina Interchange
# V6: Sion Circle Interchange
# V7: Senapati Bapat Marg / Lower Parel
# V8: Bandra-Worli Sea Link / CST South Terminal (Exit)

NODE_NAMES = [
    "V1_Borivali_Toll",
    "V2_JVLR_Interchange",
    "V3_Dadar_TT_Circle",
    "V4_Jogeshwari_WEH",
    "V5_BKC_Connector",
    "V6_Sion_Circle",
    "V7_Lower_Parel",
    "V8_South_Terminal"
]

# Directed Edges (src -> dst):
# 0: V1 -> V2
# 1: V1 -> V4
# 2: V1 -> V5
# 3: V2 -> V3
# 4: V2 -> V5
# 5: V3 -> V8
# 6: V4 -> V5
# 7: V5 -> V6
# 8: V6 -> V2
# 9: V6 -> V7
# 10: V7 -> V3

EDGES = [
    (0, 1), # E0: V1 -> V2
    (0, 3), # E1: V1 -> V4
    (0, 4), # E2: V1 -> V5
    (1, 2), # E3: V2 -> V3
    (1, 4), # E4: V2 -> V5
    (2, 7), # E5: V3 -> V8
    (3, 4), # E6: V4 -> V5
    (4, 5), # E7: V5 -> V6
    (5, 1), # E8: V6 -> V2 (Feedback Loop)
    (5, 6), # E9: V6 -> V7
    (6, 2)  # E10: V7 -> V3
]

EDGE_NAMES = [f"E{idx}_V{src+1}->V{dst+1}" for idx, (src, dst) in enumerate(EDGES)]

# Base Edge Physical Properties
# [Capacity (veh/hr), Length (km), Base PCI (0-100), Speed Limit (km/h), Lanes]
EDGE_PROPERTIES = np.array([
    [4000, 3.5, 85, 70, 4], # E0: V1->V2 (WEH Main)
    [3200, 2.8, 80, 60, 3], # E1: V1->V4
    [3000, 4.5, 75, 60, 3], # E2: V1->V5
    [3500, 4.0, 78, 60, 4], # E3: V2->V3
    [2800, 1.8, 82, 50, 3], # E4: V2->V5
    [4200, 2.5, 90, 80, 4], # E5: V3->V8 (Sea Link Corridor)
    [2500, 1.5, 70, 50, 2], # E6: V4->V5
    [3600, 2.0, 75, 50, 4], # E7: V5->V6 (BKC-Sion Link)
    [2400, 2.2, 72, 50, 2], # E8: V6->V2 (Loop)
    [2800, 3.0, 76, 50, 3], # E9: V6->V7
    [3100, 2.1, 80, 50, 3], # E10: V7->V3
])

# Junction Types (0: Entry/Highway, 1: Signalized, 2: Heavy Interchange/Roundabout)
NODE_TYPES = np.array([0, 2, 2, 1, 2, 2, 1, 0])

# Spatial Coordinates (Normalized approx relative lat/long for Mumbai)
NODE_COORDS = np.array([
    [0.20, 0.90], # V1 Borivali
    [0.35, 0.70], # V2 JVLR
    [0.45, 0.35], # V3 Dadar
    [0.28, 0.75], # V4 Jogeshwari
    [0.40, 0.55], # V5 BKC
    [0.52, 0.50], # V6 Sion
    [0.42, 0.30], # V7 Lower Parel
    [0.50, 0.15]  # V8 South Terminal
])

def generate_dataset():
    print(f"Generating 10,000-step Mumbai Traffic Dataset for 8 Nodes & 11 Edges...")
    
    # 1. Temporal Features Generation
    # Timestamps starting 2026-06-01 (Monsoon Season in Mumbai)
    timestamps = pd.date_range(start="2026-06-01 00:00:00", periods=NUM_TIMESTEPS, freq=f"{TIME_INTERVAL_MINS}min")
    
    hours = (timestamps.hour + timestamps.minute / 60.0).to_numpy()
    day_of_week = timestamps.dayofweek.to_numpy()
    is_weekend = (day_of_week >= 5).astype(int)
    
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    
    # Mumbai Holidays / Special Events (e.g. IPL match, monsoon alert days)
    holiday_flag = np.zeros(NUM_TIMESTEPS, dtype=int)
    special_event = np.zeros(NUM_TIMESTEPS, dtype=int)
    
    # Mark specific weekend peak days as special events
    special_event[is_weekend == 1] = np.random.choice([0, 1], size=np.sum(is_weekend), p=[0.7, 0.3])
    
    temporal_df = pd.DataFrame({
        "timestamp": timestamps,
        "hour": hours,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "holiday_flag": holiday_flag,
        "special_event": special_event
    })
    
    # 2. Weather Features Generation (Mumbai Monsoon Profile)
    # Monsoon rain bursts: periodic heavy downpours
    base_rain = np.maximum(0, np.sin(np.linspace(0, 15 * np.pi, NUM_TIMESTEPS)) * 40)
    rain_spikes = np.random.exponential(scale=15, size=NUM_TIMESTEPS) * (np.random.rand(NUM_TIMESTEPS) > 0.85)
    precip_intensity = np.clip(base_rain + rain_spikes, 0, 120.0) # mm/hr
    
    # Visibility drops as rain increases
    visibility = np.clip(5000 - precip_intensity * 35 + np.random.normal(0, 100, NUM_TIMESTEPS), 200, 5000) # meters
    temperature = 28.0 + 4.0 * np.sin(2 * np.pi * (hours - 8) / 24.0) + np.random.normal(0, 0.5, NUM_TIMESTEPS) # deg C
    
    # 3. Base Demand Curve (Diurnal Traffic Flow for Mumbai)
    # Morning Peak: 8:00 AM - 11:30 AM (Inbound South/BKC)
    morning_peak = 1.0 * np.exp(-((hours - 9.5) ** 2) / (2 * 1.5 ** 2))
    # Evening Peak: 5:00 PM - 9:00 PM (Outbound North)
    evening_peak = 1.1 * np.exp(-((hours - 18.5) ** 2) / (2 * 2.0 ** 2))
    # Midday plateau
    midday = 0.4 * np.exp(-((hours - 14.0) ** 2) / (2 * 2.5 ** 2))
    
    base_demand = 0.2 + morning_peak + evening_peak + midday
    base_demand = np.clip(base_demand, 0.15, 1.25)
    
    # Weekend reduction
    base_demand[is_weekend == 1] *= 0.65
    
    # 4. Synthesize Node & Edge Data over Time
    node_records = []
    edge_records = []
    target_records = []
    
    for t in range(NUM_TIMESTEPS):
        rain_t = precip_intensity[t]
        vis_t = visibility[t]
        temp_t = temperature[t]
        demand_t = base_demand[t]
        
        # Weather capacity degradation factor (Monsoon effect)
        rain_degradation = np.clip(1.0 - (rain_t / 150.0), 0.4, 1.0)
        
        # --- Node States ---
        node_flows = []
        node_speeds = []
        node_occupancies = []
        node_queues = []
        
        for n in range(8):
            # Demand variation per node location
            node_mult = 1.2 if n in [1, 2, 4, 5] else 0.85 # Hubs carry more flow
            flow = demand_t * 2800 * node_mult + np.random.normal(0, 100)
            flow = np.clip(flow, 200, 4200)
            
            # Greenshields Speed-Density relationship with rain degradation
            free_speed = 60.0 if n in [0, 7] else 45.0
            speed = free_speed * (1.0 - (flow / 4500.0) * 0.7) * rain_degradation + np.random.normal(0, 2)
            speed = np.clip(speed, 8.0, free_speed)
            
            occupancy = np.clip((flow / 4500.0) * 0.85 / rain_degradation + np.random.normal(0, 0.02), 0.05, 0.98)
            queue_len = np.clip((1.0 - speed / free_speed) * 400 + (rain_t * 1.5) + np.random.normal(0, 15), 0, 450)
            
            node_flows.append(flow)
            node_speeds.append(speed)
            node_occupancies.append(occupancy)
            node_queues.append(queue_len)
            
            node_records.append({
                "timestamp_idx": t,
                "node_idx": n,
                "node_name": NODE_NAMES[n],
                "flow": round(flow, 2),
                "speed": round(speed, 2),
                "occupancy": round(occupancy, 4),
                "queue_length": round(queue_len, 2),
                "junction_type": NODE_TYPES[n],
                "coord_x": NODE_COORDS[n, 0],
                "coord_y": NODE_COORDS[n, 1],
                "precip_intensity": round(rain_t, 2),
                "visibility": round(vis_t, 2),
                "temperature": round(temp_t, 2)
            })
            
        # --- Edge States & Targets ---
        # Incident probability increases during heavy rain & peak hours
        incident_prob = 0.02 + 0.15 * (rain_t > 40.0) + 0.05 * (demand_t > 0.8)
        
        edge_flows_t = []
        edge_speeds_t = []
        
        for e_idx, (src, dst) in enumerate(EDGES):
            cap, length, base_pci, spd_limit, lanes = EDGE_PROPERTIES[e_idx]
            
            # Dynamic PCI reduction under heavy rain
            current_pci = np.clip(base_pci - (rain_t * 0.25), 30, 100)
            
            # Incident flag
            has_incident = int(np.random.rand() < incident_prob)
            eff_cap = cap * rain_degradation * (0.5 if has_incident else 1.0)
            
            # Link Flow derived from source/target nodes + noise
            link_flow = 0.5 * (node_flows[src] + node_flows[dst]) * (eff_cap / cap) + np.random.normal(0, 50)
            link_flow = np.clip(link_flow, 150, eff_cap * 1.1)
            
            # Link Speed via BPR Function
            # tau = tau0 * (1 + alpha * (V / C)^beta)
            v_c_ratio = link_flow / (eff_cap + 1e-5)
            bpr_delay = 1.0 + 0.15 * (v_c_ratio ** 4)
            link_speed = np.clip(spd_limit / bpr_delay + np.random.normal(0, 2), 5.0, spd_limit)
            
            edge_flows_t.append(link_flow)
            edge_speeds_t.append(link_speed)
            
            edge_records.append({
                "timestamp_idx": t,
                "edge_idx": e_idx,
                "edge_name": EDGE_NAMES[e_idx],
                "src_node": src,
                "dst_node": dst,
                "capacity": cap,
                "effective_capacity": round(eff_cap, 2),
                "length": length,
                "pci": round(current_pci, 2),
                "speed_limit": spd_limit,
                "lanes": lanes,
                "incident_flag": has_incident,
                "flow": round(link_flow, 2),
                "speed": round(link_speed, 2)
            })
            
    # Convert lists to DataFrames
    df_nodes = pd.DataFrame(node_records)
    df_edges = pd.DataFrame(edge_records)
    
    # 5. Create Target Variables (t+1 forecast)
    print("Computing t+1 Target Labels for Edge Flow & Speed...")
    
    # Group edges by edge_idx and shift by -1 for t+1 prediction
    df_edges["target_next_flow"] = df_edges.groupby("edge_idx")["flow"].shift(-1)
    df_edges["target_next_speed"] = df_edges.groupby("edge_idx")["speed"].shift(-1)
    
    # Fill the last step (t=9999) with current value
    df_edges["target_next_flow"] = df_edges["target_next_flow"].ffill().bfill()
    df_edges["target_next_speed"] = df_edges["target_next_speed"].ffill().bfill()
    
    # 6. Save Files in Dataset Directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving dataset files to Dataset/ directory...")
    
    df_nodes.to_csv(os.path.join(output_dir, "mumbai_nodes_10k.csv"), index=False)
    df_edges.to_csv(os.path.join(output_dir, "mumbai_edges_10k.csv"), index=False)
    temporal_df.to_csv(os.path.join(output_dir, "mumbai_temporal_10k.csv"), index=False)
    
    # Save Graph Topology JSON Metadata
    graph_meta = {
        "num_nodes": 8,
        "num_edges": 11,
        "num_timesteps": NUM_TIMESTEPS,
        "nodes": [{"id": i, "name": NODE_NAMES[i], "type": int(NODE_TYPES[i]), "coords": NODE_COORDS[i].tolist()} for i in range(8)],
        "edges": [{"id": idx, "src": src, "dst": dst, "name": EDGE_NAMES[idx]} for idx, (src, dst) in enumerate(EDGES)],
        "zones": {
            "Zone_1_North": [0, 3],
            "Zone_2_Central_Loop": [1, 4, 5],
            "Zone_3_South": [2, 6, 7]
        }
    }
    
    with open(os.path.join(output_dir, "graph_topology.json"), "w") as f:
        json.dump(graph_meta, f, indent=4)
        
    # Build 3D Tensor .npz file for easy loading in PyTorch / PennyLane
    print("Building compressed .npz Tensor archive for fast PyTorch loading...")
    
    # Reshape arrays:
    # node_features: (10000, 8, 6) -> [flow, speed, occupancy, queue_length, temp, rain]
    X_node = df_nodes[["flow", "speed", "occupancy", "queue_length", "temperature", "precip_intensity"]].values.reshape(NUM_TIMESTEPS, 8, -1)
    
    # edge_features: (10000, 11, 6) -> [capacity, length, pci, speed_limit, lanes, incident_flag]
    E_edge = df_edges[["capacity", "length", "pci", "speed_limit", "lanes", "incident_flag"]].values.reshape(NUM_TIMESTEPS, 11, -1)
    
    # temporal_features: (10000, 4) -> [hour_sin, hour_cos, is_weekend, special_event]
    T_temp = temporal_df[["hour_sin", "hour_cos", "is_weekend", "special_event"]].values
    
    # targets: (10000, 11) -> next flow on edges
    Y_target_flow = df_edges["target_next_flow"].values.reshape(NUM_TIMESTEPS, 11)
    Y_target_speed = df_edges["target_next_speed"].values.reshape(NUM_TIMESTEPS, 11)
    
    np.savez_compressed(
        os.path.join(output_dir, "mumbai_stqgcn_dataset_10k.npz"),
        node_features=X_node,
        edge_features=E_edge,
        temporal_features=T_temp,
        target_flow=Y_target_flow,
        target_speed=Y_target_speed
    )
    
    print("Dataset generation complete! All files successfully created in Dataset/.")

if __name__ == "__main__":
    generate_dataset()
