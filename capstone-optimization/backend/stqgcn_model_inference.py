"""
stqgcn_model_inference.py
========================
Real STQGCN model inference for traffic prediction on the 8-node Mumbai network.
Loads a trained model and runs actual predictions instead of heuristic redistribution.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Quantum VQC Circuit Configuration (using lightweight deterministic approximation) ---
NUM_QUBITS = 8
NUM_LAYERS = 3  # K=3 (middle ground between speed and accuracy)

# Directed Graph Edges (Unidirectional, 8-node Mumbai network)
EDGES = [
    (0, 1), (0, 3), (0, 4),  # V1 -> V2, V4, V5
    (1, 2), (1, 4),          # V2 -> V3, V5
    (2, 7),                  # V3 -> V8
    (3, 4),                  # V4 -> V5
    (4, 5),                  # V5 -> V6
    (5, 1), (5, 6),          # V6 -> V2, V7
    (6, 2)                   # V7 -> V3
]

NUM_NODES = 8
NUM_EDGES = len(EDGES)

# Normalization constants (from training dataset)
MU_TRAFFIC = 1541.9044
STD_TRAFFIC = 953.3633

# Node and edge feature normalization (computed from training set) - as float32
X_MEAN = np.array([1759.7187, 31.7935, 0.3767, 164.1562, 27.9942, 15.8718], dtype=np.float32)
X_STD = np.array([1133.5748, 10.3626, 0.2471, 89.0128, 2.8832, 17.4216], dtype=np.float32) + 1e-6

E_MEAN = np.array([3190.9091, 2.7182, 74.4866, 57.2727, 3.1818, 0.0430], dtype=np.float32)
E_STD = np.array([555.0668, 0.9074, 7.0414, 9.6209, 0.7158, 0.2028], dtype=np.float32) + 1e-6

# Note: For production, replace this with actual PennyLane quantum circuits.
# The current implementation uses a deterministic approximation for compatibility.

class QuantumVQCModule(nn.Module):
    """Lightweight Quantum VQC module - deterministic approximation."""
    def __init__(self, n_layers=3, n_qubits=8):
        super().__init__()
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )
    
    def forward(self, angles):
        """
        Batch inference over angles using a lightweight deterministic approximation.
        Instead of running full quantum circuits, use a simple deterministic model
        that mimics VQC behavior without PennyLane's measurement overhead.
        """
        batch_size = angles.size(0)
        weights_cpu = self.weights.cpu().detach().numpy()
        
        # Simple deterministic approximation: angles affect output as sin/cos combinations
        out_list = []
        for i in range(batch_size):
            angle_i = angles[i].cpu().detach().numpy()
            # Mimic quantum behavior with trigonometric functions
            expvals_approx = []
            for j in range(self.n_qubits):
                # Combine angle with learned weights for each qubit
                combined = angle_i[j] + np.sum(weights_cpu[:, j, :])
                # Use tanh to get values in [-1, 1] range (similar to Pauli-Z expectation)
                expval = np.tanh(combined)
                expvals_approx.append(float(expval))
            out_list.append(expvals_approx)
        
        expvals = torch.from_numpy(np.array(out_list, dtype=np.float32)).float()
        return expvals

# --- STQGCN Model Definition ---
class STQGCN(nn.Module):
    def __init__(self, num_nodes, num_edges, node_in_dim=6, edge_in_dim=6, 
                 temp_in_dim=4, hidden_dim=16, n_qubits=8, n_vqc_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.edges = EDGES
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.n_vqc_layers = n_vqc_layers
        
        # 1. Node Embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim + temp_in_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. Message Passing
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_in_dim + temp_in_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 3. Quantum Pre-Projection
        self.q_pre_proj = nn.Sequential(
            nn.Linear(hidden_dim + temp_in_dim, 1),
            nn.Tanh()
        )
        
        # 4. Quantum VQC
        self.q_vqc = QuantumVQCModule(n_layers=n_vqc_layers, n_qubits=n_qubits)
        
        # 5. Quantum Post-Projection
        self.q_post_proj = nn.Sequential(
            nn.Linear(1 + temp_in_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 6. Edge Prediction Head
        self.pred_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in_dim + temp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, X_node, E_edge, T_temp):
        B = X_node.size(0)
        
        # Step 1: Node Embedding
        T_expanded = T_temp.unsqueeze(1).expand(-1, self.num_nodes, -1)
        node_inputs = torch.cat([X_node, T_expanded], dim=-1)
        h_0 = self.node_embed(node_inputs)
        
        # Step 2: Message Passing
        a_aggregated = torch.zeros_like(h_0)
        for e_idx, (src, dst) in enumerate(self.edges):
            h_src = h_0[:, src, :]
            e_feat = E_edge[:, e_idx, :]
            t_feat = T_temp
            msg_input = torch.cat([h_src, e_feat, t_feat], dim=-1)
            msg = self.message_mlp(msg_input)
            a_aggregated[:, dst, :] += msg
        
        # Step 3: Residual Update
        h_1 = h_0 + a_aggregated
        
        # Step 4: Quantum Interface
        node_angles = self.q_pre_proj(torch.cat([h_1, T_expanded], dim=-1)).squeeze(-1) * np.pi
        
        angles_cpu = node_angles.cpu()
        q_expval_cpu = self.q_vqc(angles_cpu)
        q_expval = q_expval_cpu.to(X_node.device).float().unsqueeze(-1)
        
        q_post = self.q_post_proj(torch.cat([q_expval, T_expanded], dim=-1))
        h_final = self.layer_norm(h_1 + q_post)
        
        # Step 5: Edge Prediction
        predictions = []
        for e_idx, (src, dst) in enumerate(self.edges):
            h_s = h_final[:, src, :]
            h_d = h_final[:, dst, :]
            e_feat = E_edge[:, e_idx, :]
            t_feat = T_temp
            pred_input = torch.cat([h_s, h_d, e_feat, t_feat], dim=-1)
            pred_out = self.pred_mlp(pred_input).squeeze(-1)
            predictions.append(pred_out)
        
        y_hat_norm = torch.stack(predictions, dim=1)
        return y_hat_norm

# --- Load Model ---
model = None
model_path = None

def load_stqgcn_model():
    """Load trained STQGCN model."""
    global model, model_path
    
    if model is not None:
        return model
    
    # Try to find the trained model
    # From backend folder: go up 2 levels to 30-August-Capstone, then into traffic-forecasting-final
    base_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        "../../traffic-forecasting-final/Traffic Forecasting/Single Network/Unidirectional/STQGCN"
    ))
    
    model_file = os.path.join(base_path, "stqgcn_K3_model.pt")
    
    if not os.path.exists(model_file):
        print(f"Warning: Model file not found at {model_file}")
        print(f"  Current working directory: {os.getcwd()}")
        print(f"  Script directory: {os.path.dirname(__file__)}")
        # Try alternative path
        alt_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../../traffic-forecasting-final/Traffic Forecasting/Single Network/Unidirectional/STQGCN/stqgcn_K3_model.pt"
        ))
        if os.path.exists(alt_path):
            model_file = alt_path
            print(f"  Found at alternative path: {model_file}")
        else:
            return None
    
    print(f"Loading STQGCN model from: {model_file}")
    model = STQGCN(
        num_nodes=NUM_NODES,
        num_edges=NUM_EDGES,
        node_in_dim=6,
        edge_in_dim=6,
        temp_in_dim=4,
        hidden_dim=16,
        n_qubits=NUM_QUBITS,
        n_vqc_layers=NUM_LAYERS
    ).to(device)
    
    state_dict = torch.load(model_file, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    model_path = model_file
    print("Model loaded successfully!")
    return model

def predict_stqgcn_traffic_real(
    current_predictions: Dict[str, float],
    edges_list: List[dict],
    network_type: str = "single8",
    changed_edge_id: str | None = None,
    new_value: float | None = None
) -> Dict[str, float]:
    """
    Run actual STQGCN model inference to predict edge traffic.
    
    Args:
        current_predictions: Current edge traffic values (dict keyed by edge ID like "V1→V2")
        edges_list: List of edge objects with id, source, target, capacity, etc.
        network_type: Network topology identifier (default: "single8")
        changed_edge_id: The edge ID that the user manually changed (kept as-is)
        new_value: The new value for changed_edge_id (kept as-is)
    
    Returns:
        Dictionary of predicted edge traffic values for all edges except the manually changed one
    """
    model = load_stqgcn_model()
    if model is None:
        print("Could not load STQGCN model, returning current predictions")
        return dict(current_predictions)  # Return a copy of current values
    
    try:
        # Create feature tensors using current predictions as historical flow
        # Node features: [historical_flow, signal_cycle_time, junction_type, x, y, degree]
        node_flows = [
            sum(v for k, v in current_predictions.items() if k.startswith("V1→")) or 630,  # V1
            sum(v for k, v in current_predictions.items() if k.startswith("V2→")) or 520,  # V2
            sum(v for k, v in current_predictions.items() if k.startswith("V3→")) or 650,  # V3
            sum(v for k, v in current_predictions.items() if k.startswith("V4→")) or 602,  # V4
            sum(v for k, v in current_predictions.items() if k.startswith("V5→")) or 717,  # V5
            sum(v for k, v in current_predictions.items() if k.startswith("V6→")) or 494,  # V6
            sum(v for k, v in current_predictions.items() if k.startswith("V7→")) or 586,  # V7
            sum(v for k, v in current_predictions.items() if k.startswith("V8→")) or 525,  # V8
        ]
        
        node_features = torch.tensor([
            [node_flows[0], 51.4, 0.10, 84.0, 25.0, 0.0],  # V1
            [node_flows[1], 37.6, 0.13, 62.6, 25.0, 0.0],  # V2
            [node_flows[2], 40.0, 0.09, 63.8, 25.0, 0.0],  # V3
            [node_flows[3], 41.6, 0.07, 46.9, 25.0, 0.0],  # V4
            [node_flows[4], 41.2, 0.15,  9.4, 25.0, 0.0],  # V5
            [node_flows[5], 39.6, 0.14, 54.3, 25.0, 0.0],  # V6
            [node_flows[6], 39.4, 0.09, 21.5, 25.0, 0.0],  # V7
            [node_flows[7], 58.1, 0.06, 14.4, 25.0, 0.0],  # V8
        ], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 8, 6)
        
        # Normalize node features
        node_features = (node_features - torch.from_numpy(X_MEAN).float().to(device)) / torch.from_numpy(X_STD).float().to(device)
        
        # Edge features: 6 dimensions matching training dataset
        edge_features = torch.tensor([
            [4000.0, 3.5, 85.0, 70.0, 4.0, 0.0],  # V1→V2
            [3200.0, 2.8, 80.0, 60.0, 3.0, 0.0],  # V1→V4
            [3000.0, 4.5, 75.0, 60.0, 3.0, 0.0],  # V1→V5
            [3500.0, 4.0, 78.0, 60.0, 4.0, 0.0],  # V2→V3
            [2800.0, 1.8, 82.0, 50.0, 3.0, 0.0],  # V2→V5
            [4200.0, 2.5, 90.0, 80.0, 4.0, 0.0],  # V3→V8
            [2500.0, 1.5, 70.0, 50.0, 2.0, 0.0],  # V4→V5
            [3600.0, 2.0, 75.0, 50.0, 4.0, 0.0],  # V5→V6
            [2400.0, 2.2, 72.0, 50.0, 2.0, 1.0],  # V6→V2
            [2800.0, 3.0, 76.0, 50.0, 3.0, 0.0],  # V6→V7
            [3100.0, 2.1, 80.0, 50.0, 3.0, 0.0],  # V7→V3
        ], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 11, 6)
        
        # Normalize edge features
        edge_features = (edge_features - torch.from_numpy(E_MEAN).float().to(device)) / torch.from_numpy(E_STD).float().to(device)
        
        # Cyclical temporal features: [sin(t), cos(t), day_type, season] in [-1, 1] range
        temporal_features = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float32).to(device)  # (1, 4)
        
        # Run inference
        with torch.no_grad():
            predictions_norm = model(node_features, edge_features, temporal_features)  # (1, 11)
        
        # Denormalize predictions
        predictions_denorm = predictions_norm.cpu().numpy()[0] * float(STD_TRAFFIC) + float(MU_TRAFFIC)
        
        # Map to edge IDs (must match EDGES list order)
        edge_id_map = [
            "V1→V2", "V1→V4", "V1→V5",
            "V2→V3", "V2→V5",
            "V3→V8",
            "V4→V5",
            "V5→V6",
            "V6→V2", "V6→V7",
            "V7→V3"
        ]
        
        result = {}
        for i, edge_id in enumerate(edge_id_map):
            # If this is the edge the user manually changed, keep their value
            if edge_id == changed_edge_id and new_value is not None:
                result[edge_id] = max(0.0, float(new_value))
            else:
                # Otherwise use the model prediction
                value = max(0.0, float(predictions_denorm[i]))
                result[edge_id] = value
        
        safe_kept_id = changed_edge_id.replace("→", "->") if changed_edge_id else ""
        kept_text = f" (kept {safe_kept_id} at {new_value} veh/hr)" if changed_edge_id else ""
        try:
            print(f"[OK] STQGCN model inference complete. Predicted 11 edges{kept_text}.")
        except Exception:
            pass

        for edge_id, pred in sorted(result.items()):
            marker = " <- kept" if edge_id == changed_edge_id else ""
            safe_id = edge_id.replace("→", "->")
            try:
                print(f"  {safe_id}: {pred:.1f} veh/hr{marker}")
            except Exception:
                pass
        
        return result
    
    except Exception as e:
        print(f"[ERROR] Error running STQGCN inference: {e}")
        import traceback
        traceback.print_exc()
        return dict(current_predictions)  # Return a copy as fallback
