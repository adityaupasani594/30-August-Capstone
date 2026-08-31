from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import pennylane as qml
import pandas as pd
import numpy as np

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =========================================================
# 1. GRAPH DEFINITION & RAW DATA
# =========================================================
# Nodes: A=0, B=1, C=2, D=3, E=4
node_mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
inv_node_mapping = {v: k for k, v in node_mapping.items()}

# Directed Edges
edges = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'D'), ('C', 'D'), ('D', 'E'), ('E', 'B')]
edge_index = torch.tensor([[node_mapping[src] for src, dst in edges],
                           [node_mapping[dst] for src, dst in edges]], dtype=torch.long)

# Raw Node Features (5 nodes, 6 features each)
# [historical_flow, signal_cycle_time, junction_type, x_coordinate, y_coordinate, node_degree]
X_raw = torch.tensor([
    [1200.0, 40.0, 1.0, 0.10, 0.60, 3.0], # A
    [800.0,  60.0, 1.0, 0.35, 0.80, 3.0], # B
    [600.0,  45.0, 1.0, 0.45, 0.50, 3.0], # C
    [1500.0, 70.0, 1.0, 0.70, 0.30, 4.0], # D
    [900.0,  50.0, 1.0, 0.90, 0.55, 2.0]  # E
], dtype=torch.float32)

# Raw Edge Features (7 edges, 5 features each)
# [capacity, speed_limit, lanes, length, road_type]
E_raw = torch.tensor([
    [1800.0, 50.0, 3.0, 1.20, 1.0], # A->B
    [1490.0, 50.0, 3.0, 0.95, 1.0], # A->C
    [2200.0, 60.0, 4.0, 1.60, 2.0], # A->D
    [1800.0, 50.0, 3.0, 1.10, 1.0], # B->D
    [1300.0, 40.0, 2.0, 0.80, 1.0], # C->D
    [2200.0, 60.0, 4.0, 1.40, 2.0], # D->E
    [1800.0, 50.0, 3.0, 1.00, 1.0]  # E->B
], dtype=torch.float32)

# Denormalization constants
traffic_mean = 1200.0
traffic_std = 450.0

# =========================================================
# 2. STEP-BY-STEP MODULES
# =========================================================

# Step 1: Z-score Normalization (per feature across all elements)
def z_score_normalize(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True, unbiased=False)
    # Add small epsilon to prevent division by zero if std is 0
    return (tensor - mean) / (std + 1e-8)

X_norm = z_score_normalize(X_raw)
E_norm = z_score_normalize(E_raw)

# Dimensions
d_v = 6
d_e = 5
d_h = 8
n_qubits = 4

# Step 2: Classical Node & Edge Embedding Layer
class ClassicalEmbedding(nn.Module):
    def __init__(self, d_v, d_e, d_h):
        super().__init__()
        self.W_v = nn.Linear(d_v, d_h)
        self.W_e = nn.Linear(d_e, d_h)
        self.relu = nn.ReLU()
        
    def forward(self, X, E):
        h = self.relu(self.W_v(X))
        f = self.relu(self.W_e(E))
        return h, f

# Step 3 & 4: Message Passing & Neighborhood Aggregation
class MessagePassingLayer(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        # MLP takes concatenated [h_j || f_ji] -> 2 * d_h inputs
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_h, 2 * d_h),
            nn.ReLU(),
            nn.Linear(2 * d_h, d_h)
        )
        
    def forward(self, h, f, edge_index):
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)
        
        # Aggregate messages at target nodes
        aggregated_messages = torch.zeros((num_nodes, d_h), dtype=torch.float32)
        
        for e in range(num_edges):
            src = edge_index[0, e].item()
            dst = edge_index[1, e].item()
            
            # Construct message from src to dst using source node and edge feature
            h_j = h[src]
            f_ji = f[e]
            msg_input = torch.cat([h_j, f_ji], dim=-1)
            msg = self.mlp(msg_input)
            
            # Sum aggregation into target node
            aggregated_messages[dst] += msg
            
        return aggregated_messages

# Step 6: Quantum Update Layer using PennyLane
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Step B: Angle Embedding using RY rotations
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Step C: Variational quantum circuit with entanglement (StronglyEntanglingLayers)
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    
    # Step D: Measurement of Pauli-Z expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumUpdate(nn.Module):
    def __init__(self, d_h, n_qubits):
        super().__init__()
        # Step A: Classical Pre-projection (8 -> 4)
        self.W_in = nn.Linear(d_h, n_qubits)
        
        # Trainable quantum weights (initialized randomly for forward pass)
        # 2 layers, 4 qubits, 3 parameters per qubit per layer
        self.q_weights = nn.Parameter(torch.rand((2, n_qubits, 3)) * 2 * np.pi)
        
        # Step E: Classical Post-projection (4 -> 8)
        self.W_out = nn.Linear(n_qubits, d_h)
        
    def forward(self, h_tilde):
        num_nodes = h_tilde.size(0)
        q_out = torch.zeros((num_nodes, d_h), dtype=torch.float32)
        
        for i in range(num_nodes):
            # Step A: Pre-projection & scaling to [-pi, pi]
            z = self.W_in(h_tilde[i])
            a = np.pi * torch.tanh(z)
            
            # Execute Quantum Circuit
            expvals = quantum_circuit(a, self.q_weights)
            o = torch.stack(expvals).float()
            
            # Step E: Post-projection back to embedding space
            q_out[i] = self.W_out(o)
            
        return q_out

# Step 8: Edge-Level Traffic Prediction MLP
class EdgePredictionMLP(nn.Module):
    def __init__(self, d_h):
        super().__init__()
        # Takes concatenated [h_i || h_j || f_ij] -> 3 * d_h inputs
        self.mlp = nn.Sequential(
            nn.Linear(3 * d_h, d_h),
            nn.ReLU(),
            nn.Linear(d_h, 1)
        )
        
    def forward(self, h, f, edge_index):
        num_edges = edge_index.size(1)
        predictions = torch.zeros(num_edges, dtype=torch.float32)
        
        for e in range(num_edges):
            src = edge_index[0, e].item()
            dst = edge_index[1, e].item()
            
            h_i = h[src]
            h_j = h[dst]
            f_ij = f[e]
            
            r_ij = torch.cat([h_i, h_j, f_ij], dim=-1)
            predictions[e] = self.mlp(r_ij)
            
        return predictions

# Complete Pipeline Assembler
class STQGCN(nn.Module):
    def __init__(self, d_v, d_e, d_h, n_qubits):
        super().__init__()
        self.embedding = ClassicalEmbedding(d_v, d_e, d_h)
        self.message_passing = MessagePassingLayer(d_h)
        self.quantum_update = QuantumUpdate(d_h, n_qubits)
        self.layer_norm = nn.LayerNorm(d_h)
        self.predict_mlp = EdgePredictionMLP(d_h)
        
    def forward(self, X, E, edge_index):
        # 1. Node/Edge initial embedding
        h_0, f = self.embedding(X, E)
        
        # 2. Message Passing & Neighborhood Aggregation
        a = self.message_passing(h_0, f, edge_index)
        
        # 3. Step 5: Residual Update
        h_tilde = h_0 + a
        
        # 4. Step 6: Quantum Processing Layer
        q = self.quantum_update(h_tilde)
        
        # 5. Step 7: Residual + LayerNorm (Corrected Form in Equation 10)
        h_final = self.layer_norm(h_tilde + q)
        
        # 6. Step 8: Edge Prediction
        y_hat = self.predict_mlp(h_final, f, edge_index)
        
        # 7. Step 9: Denormalization
        y_actual = y_hat * traffic_std + traffic_mean
        return y_actual

# =========================================================
# 3. PIPELINE EXECUTION & OUTPUTS
# =========================================================
model = STQGCN(d_v, d_e, d_h, n_qubits)
model.eval()

with torch.no_grad():
    predicted_congestion = model(X_norm, E_norm, edge_index).numpy()

# Extract statistics
min_congestion = float(predicted_congestion.min())
max_congestion = float(predicted_congestion.max())
avg_congestion = float(predicted_congestion.mean())

# Formatted console outputs
print("Predicted Edge Congestion (veh/hr)")
edge_labels = []
for idx, (src, dst) in enumerate(edges):
    label = f"{src}→{dst}"
    edge_labels.append(label)
    print(f"{label} : {predicted_congestion[idx]:.2f}")

print(f"\nMinimum predicted congestion : {min_congestion:.2f}")
print(f"Maximum predicted congestion : {max_congestion:.2f}")
print(f"Average predicted congestion : {avg_congestion:.2f}")

# Create Pandas Dataframe output
df_results = pd.DataFrame({
    'Edge': edge_labels,
    'Predicted Congestion (veh/hr)': [round(val, 2) for val in predicted_congestion]
})


def predict_stqgcn_traffic(
    predictions: Dict[str, float],
    changed_edge_id: str | None = None,
    new_value: float | None = None,
    edges: Iterable[dict] | None = None,
    network_type: str = "single8",
) -> Dict[str, float]:
    """Return a deterministic STQGCN-inspired reforecast for the current network.

    This keeps the backend compatible with the frontend interaction while staying
    lightweight enough for a demo app. The logic preserves the existing graph
    structure and redistributes traffic on neighbouring edges based on their
    current load and capacity.
    """
    current = dict(predictions or {})
    if not current:
        return {}

    edge_list = list(edges or [])
    if edge_list and changed_edge_id and new_value is not None:
        edited = next((edge for edge in edge_list if edge.get("id") == changed_edge_id), None)
        if edited:
            current[changed_edge_id] = max(0.0, float(new_value))
            connected_ids = [
                edge["id"] for edge in edge_list
                if edge.get("id") != changed_edge_id and (
                    edge.get("source") == edited.get("source")
                    or edge.get("target") == edited.get("source")
                    or edge.get("source") == edited.get("target")
                    or edge.get("target") == edited.get("target")
                )
            ]

            if connected_ids:
                total_connected = sum(float(current.get(edge_id, 0.0)) for edge_id in connected_ids)
                for edge_id in connected_ids:
                    weight = (float(current.get(edge_id, 0.0)) / total_connected) if total_connected > 0 else 1.0 / len(connected_ids)
                    delta = (float(new_value) - float(predictions.get(changed_edge_id, 0.0))) * 0.65 * weight
                    current[edge_id] = max(0.0, float(current.get(edge_id, 0.0)) + delta)

    # Apply a mild global smoothing step so the graph remains physically plausible.
    total = sum(float(value) for value in current.values())
    if total > 0.0:
        for edge_id, value in list(current.items()):
            edge = next((e for e in edge_list if e.get("id") == edge_id), None)
            capacity = float(edge.get("capacity", 0.0)) if edge else 0.0
            upper_bound = capacity * 1.20 if capacity > 0 else 2200.0
            current[edge_id] = max(0.0, min(float(value), upper_bound))

    return current


if __name__ == "__main__":
    pass
