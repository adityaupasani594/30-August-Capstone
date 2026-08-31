import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pennylane as qml

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {device}")

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../Dataset/Hierarchical Network/mumbai_stqgcn_hierarchical_24k.npz"))
TOPOLOGY_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../Dataset/Hierarchical Network/hierarchical_graph_topology.json"))

# Output CSV path for unified ablation metrics
ABLATION_CSV_PATH = os.path.join(SCRIPT_DIR, "hierarchical_stqgcn_vqc_ablation_metrics.csv")

# --- Load Dataset & Topology ---
print(f"Loading master dataset from: {DATASET_PATH}")
dataset = np.load(DATASET_PATH)

X_node = dataset["node_features"]       # (10000, 24, 6)
E_edge = dataset["edge_features"]       # (10000, 30, 6)
T_temp = dataset["temporal_features"]   # (10000, 4)
Y_target_raw = dataset["target_flow"]   # (10000, 30)

num_timesteps, num_nodes, node_feat_dim = X_node.shape
_, num_edges, edge_feat_dim = E_edge.shape
_, temp_feat_dim = T_temp.shape

with open(TOPOLOGY_PATH, "r") as f:
    topo_data = json.load(f)

EDGES = [(e["src"], e["dst"]) for e in topo_data["edges"]]
ZONE_DICT = topo_data["zones"]

ZONE_1_NODES = ZONE_DICT["Zone_1_Tree_North"]      # 0..7
ZONE_2_NODES = ZONE_DICT["Zone_2_Mesh_Central"]    # 8..15
ZONE_3_NODES = ZONE_DICT["Zone_3_Linear_South"]    # 16..23
NUM_QUBITS = 3 # 1 Qubit per Zone

print(f"Loaded Hierarchical Topology: {num_nodes} Nodes, {num_edges} Directed Edges across {NUM_QUBITS} Zones/Qubits")

# Calculate Normalization Statistics
mu_traffic = float(np.mean(Y_target_raw))
std_traffic = float(np.std(Y_target_raw))
print(f"Traffic Forecasting (5s Lookahead) Target Normalization Stats: mean = {mu_traffic:.2f}, std = {std_traffic:.2f}")

# Normalize Features
X_mean, X_std = np.mean(X_node, axis=(0, 1)), np.std(X_node, axis=(0, 1)) + 1e-6
E_mean, E_std = np.mean(E_edge, axis=(0, 1)), np.std(E_edge, axis=(0, 1)) + 1e-6

X_node_norm = (X_node - X_mean) / X_std
E_edge_norm = (E_edge - E_mean) / E_std
Y_target_norm = (Y_target_raw - mu_traffic) / std_traffic

# --- 5-Second Lookahead Forecasting Target Shift (Features at t -> Target at t+1) ---
X_features = X_node_norm[:-1]
E_features = E_edge_norm[:-1] if 'E_edge_norm' in locals() else input_features[:-1]
T_features = T_temp[:-1]
Y_targets = Y_target_norm[1:]
Y_raw_targets = Y_target[1:] if 'Y_target' in locals() else (Y_target_raw[1:] if 'Y_target_raw' in locals() else (Y_target_bi[1:] if 'Y_target_bi' in locals() else Y_target_norm[1:]))

num_samples = len(X_features)
train_size = int(0.70 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

X_train, X_val, X_test = X_features[:train_size], X_features[train_size:train_size+val_size], X_features[train_size+val_size:]
E_train, E_val, E_test = E_features[:train_size], E_features[train_size:train_size+val_size], E_features[train_size+val_size:]
T_train, T_val, T_test = T_features[:train_size], T_features[train_size:train_size+val_size], T_features[train_size+val_size:]
Y_train, Y_val, Y_test = Y_targets[:train_size], Y_targets[train_size:train_size+val_size], Y_targets[train_size+val_size:]
Y_raw_test = Y_raw_targets[train_size+val_size:]

batch_size = 64
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(E_train), torch.FloatTensor(T_train), torch.FloatTensor(Y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(E_val), torch.FloatTensor(T_val), torch.FloatTensor(Y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(E_test), torch.FloatTensor(T_test), torch.FloatTensor(Y_raw_test))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 3-Qubit Vectorized PennyLane QNode Device & Builder ---
q_dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(q_dev, interface="torch", diff_method="backprop")
def vqc_circuit_vectorized(inputs, weights):
    n_layers = weights.shape[0]
    
    # 1. Angle Embedding (Eq 9) for 3 Zone Qubits
    for q in range(NUM_QUBITS):
        qml.RY(inputs[:, q], wires=q)
        
    # 2. Variational Evolution & Inter-Zone CNOT Ring Entanglement (Eq 10-11)
    for k in range(n_layers):
        for q in range(NUM_QUBITS):
            qml.RZ(weights[k, q, 0], wires=q)
            qml.RY(weights[k, q, 1], wires=q)
            qml.RZ(weights[k, q, 2], wires=q)
            
        for q in range(NUM_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % NUM_QUBITS])
            
    # 3. Pauli-Z Expectation Measurement across 3 Zone Qubits
    return [qml.expval(qml.PauliZ(q)) for q in range(NUM_QUBITS)]

# Ultra-Fast PyTorch 3-Qubit Quantum Module
class QuantumVQCModule3Qubit(nn.Module):
    def __init__(self, n_layers):
        super(QuantumVQCModule3Qubit, self).__init__()
        self.n_layers = n_layers
        self.weights = nn.Parameter(torch.randn(n_layers, NUM_QUBITS, 3) * 0.1)
        
    def forward(self, angles):
        weights_cpu = self.weights.cpu()
        out_list = vqc_circuit_vectorized(angles, weights_cpu)
        expvals = torch.stack(out_list, dim=1) # (B, 3)
        return expvals

# Hierarchical STQGCN Model Definition (24 Nodes -> 3 Zone Qubits)
class HierarchicalSTQGCN(nn.Module):
    def __init__(self, num_nodes, num_edges, node_in_dim, edge_in_dim, temp_in_dim, hidden_dim=16, n_vqc_layers=2):
        super(HierarchicalSTQGCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.edges = EDGES
        self.hidden_dim = hidden_dim
        self.n_vqc_layers = n_vqc_layers
        
        # Tier 1: Micro Node Embedding
        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim + temp_in_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Tier 1: Micro Spatio-Temporal Message Passing
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_in_dim + temp_in_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Tier 2: Quantum Angle Pre-Projection for 3 Zones
        self.q_pre_proj = nn.Sequential(
            nn.Linear(hidden_dim + temp_in_dim, 1),
            nn.Tanh()
        )
        
        # 3-Qubit Vectorized VQC Module
        self.q_vqc = QuantumVQCModule3Qubit(n_layers=n_vqc_layers)
        
        # Tier 2: Quantum Post-Projection for 3 Zones
        self.q_post_proj = nn.Sequential(
            nn.Linear(1 + temp_in_dim, hidden_dim),
            nn.ReLU()
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Edge Prediction Head (for 30 edges)
        self.pred_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in_dim + temp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, X_node, E_edge, T_temp):
        B = X_node.size(0)
        
        # --- Tier 1: Micro Node Embedding & Message Passing ---
        T_expanded = T_temp.unsqueeze(1).expand(-1, self.num_nodes, -1)
        node_inputs = torch.cat([X_node, T_expanded], dim=-1)
        h_0 = self.node_embed(node_inputs)
        
        a_aggregated = torch.zeros_like(h_0)
        for e_idx, (src, dst) in enumerate(self.edges):
            h_src = h_0[:, src, :]
            e_feat = E_edge[:, e_idx, :]
            t_feat = T_temp
            
            msg_input = torch.cat([h_src, e_feat, t_feat], dim=-1)
            msg = self.message_mlp(msg_input)
            a_aggregated[:, dst, :] += msg
            
        h_1 = h_0 + a_aggregated
        
        # --- Tier 2: Macro Zone Spatial Pooling (24 Nodes -> 3 Zone Super-Nodes) ---
        z1_embed = h_1[:, ZONE_1_NODES, :].mean(dim=1, keepdim=True) # (B, 1, hidden_dim)
        z2_embed = h_1[:, ZONE_2_NODES, :].mean(dim=1, keepdim=True) # (B, 1, hidden_dim)
        z3_embed = h_1[:, ZONE_3_NODES, :].mean(dim=1, keepdim=True) # (B, 1, hidden_dim)
        
        zones_embed = torch.cat([z1_embed, z2_embed, z3_embed], dim=1) # (B, 3, hidden_dim)
        T_zone_expanded = T_temp.unsqueeze(1).expand(-1, 3, -1)
        
        # Pre-projection to 3 Quantum Rotation Angles
        zone_angles = self.q_pre_proj(torch.cat([zones_embed, T_zone_expanded], dim=-1)).squeeze(-1) * np.pi # (B, 3)
        
        # 3-Qubit PennyLane VQC Execution
        angles_cpu = zone_angles.cpu()
        q_expval_cpu = self.q_vqc(angles_cpu)
        q_expval = q_expval_cpu.to(X_node.device).float().unsqueeze(-1) # (B, 3, 1)
        
        # Post-projection back to hidden dimension
        q_post = self.q_post_proj(torch.cat([q_expval, T_zone_expanded], dim=-1)) # (B, 3, hidden_dim)
        
        # Broadcast updated quantum zone context back to Tier 1 nodes
        h_zone_broadcast = torch.zeros_like(h_1)
        h_zone_broadcast[:, ZONE_1_NODES, :] = q_post[:, 0, :].unsqueeze(1)
        h_zone_broadcast[:, ZONE_2_NODES, :] = q_post[:, 1, :].unsqueeze(1)
        h_zone_broadcast[:, ZONE_3_NODES, :] = q_post[:, 2, :].unsqueeze(1)
        
        h_final = self.layer_norm(h_1 + h_zone_broadcast)
        
        # --- Edge Prediction Head ---
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

# --- Main Ablation Experiment Loop for Hierarchical Network (K = 1, 2, 3, 4, 5, 6, 7) ---
VQC_LAYERS_TO_TEST = [1, 2, 3, 4, 5, 6, 7]
NUM_EPOCHS = 40

all_ablation_results = []
existing_keys = set()

if os.path.exists(ABLATION_CSV_PATH):
    try:
        existing_df = pd.read_csv(ABLATION_CSV_PATH)
        all_ablation_results = existing_df.to_dict("records")
        existing_keys = set(existing_df["vqc_layers_K"].values)
        print(f"Loaded {len(all_ablation_results)} existing ablation results from CSV.")
    except Exception as e:
        print(f"Could not load existing ablation CSV: {e}")

print("=================================================================")
print("STARTING HIERARCHICAL STQGCN VQC LAYER ABLATION (K = 1, 2, 3, 4, 5, 6, 7)")
print("=================================================================\n")

for k_layers in VQC_LAYERS_TO_TEST:
    if k_layers in existing_keys:
        print(f"--> Skipping K = {k_layers} (already evaluated in {ABLATION_CSV_PATH})")
        continue
        
    print(f"\n>>> Running Hierarchical STQGCN Training for K = {k_layers} VQC Layer(s) (Quantum Params: {k_layers * NUM_QUBITS * 3}) <<<")
    
    model_save_path_k = os.path.join(SCRIPT_DIR, f"hierarchical_stqgcn_K{k_layers}_model.pt")
    history_csv_k = os.path.join(SCRIPT_DIR, f"hierarchical_stqgcn_K{k_layers}_epoch_history.csv")
    
    model = HierarchicalSTQGCN(
        num_nodes=num_nodes,
        num_edges=num_edges,
        node_in_dim=node_feat_dim,
        edge_in_dim=edge_feat_dim,
        temp_in_dim=temp_feat_dim,
        hidden_dim=16,
        n_vqc_layers=k_layers
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    torch.save(model.state_dict(), model_save_path_k)

    history = []
    start_total_time = time.time()
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # Train Phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_E, batch_T, batch_Y in train_loader:
            batch_X, batch_E, batch_T, batch_Y = batch_X.to(device), batch_E.to(device), batch_T.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_X, batch_E, batch_T)
            loss = criterion(preds, batch_Y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= train_size
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_E, batch_T, batch_Y in val_loader:
                batch_X, batch_E, batch_T, batch_Y = batch_X.to(device), batch_E.to(device), batch_T.to(device), batch_Y.to(device)
                preds = model(batch_X, batch_E, batch_T)
                loss = criterion(preds, batch_Y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= val_size
        epoch_duration = time.time() - epoch_start
        
        history.append({
            "epoch": epoch,
            "train_loss_mse": round(train_loss, 6),
            "val_loss_mse": round(val_loss, 6),
            "epoch_time_sec": round(epoch_duration, 4)
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path_k)
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [K={k_layers}] Epoch [{epoch:02d}/{NUM_EPOCHS}] - Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | Time: {epoch_duration:.2f}s")

    total_training_time = time.time() - start_total_time
    
    # Save Epoch History CSV
    pd.DataFrame(history).to_csv(history_csv_k, index=False)
    
    # Test Evaluation
    model.load_state_dict(torch.load(model_save_path_k))
    model.eval()

    y_preds_raw_list = []
    y_true_raw_list = []

    with torch.no_grad():
        for batch_X, batch_E, batch_T, batch_Y_raw in test_loader:
            batch_X, batch_E, batch_T = batch_X.to(device), batch_E.to(device), batch_T.to(device)
            preds_norm = model(batch_X, batch_E, batch_T).cpu().numpy()
            preds_raw = preds_norm * std_traffic + mu_traffic
            
            y_preds_raw_list.append(preds_raw)
            y_true_raw_list.append(batch_Y_raw.numpy())

    y_pred_all = np.vstack(y_preds_raw_list)
    y_true_all = np.vstack(y_true_raw_list)

    mae = float(np.mean(np.abs(y_true_all - y_pred_all)))
    rmse = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
    mape = float(np.mean(np.abs((y_true_all - y_pred_all) / (y_true_all + 1e-5))) * 100.0)

    ss_res = np.sum((y_true_all - y_pred_all) ** 2)
    ss_tot = np.sum((y_true_all - np.mean(y_true_all)) ** 2)
    r2_score = float(1.0 - (ss_res / ss_tot))

    print(f"\n--> Test Results for Hierarchical STQGCN K={k_layers}: MAE={mae:.2f} | RMSE={rmse:.2f} | MAPE={mape:.2f}% | R2={r2_score:.4f} | Time={total_training_time:.2f}s\n")

    all_ablation_results.append({
        "model_name": f"STQGCN_Unidirectional_Hierarchical_K{k_layers}",
        "network_type": "Hierarchical_Network_Unidirectional",
        "scale": "24_Nodes_3_Zones",
        "vqc_layers_K": k_layers,
        "num_qubits": NUM_QUBITS,
        "quantum_params_theta": k_layers * NUM_QUBITS * 3,
        "mae_veh_hr": round(mae, 4),
        "rmse_veh_hr": round(rmse, 4),
        "mape_percent": round(mape, 4),
        "r2_score": round(r2_score, 4),
        "best_val_mse": round(best_val_loss, 6),
        "total_train_time_sec": round(total_training_time, 2),
        "avg_epoch_time_sec": round(total_training_time / NUM_EPOCHS, 4)
    })

# Save Unified Ablation Results CSV
ablation_df = pd.DataFrame(all_ablation_results)
ablation_df.to_csv(ABLATION_CSV_PATH, index=False)

print("=================================================================")
print("ALL HIERARCHICAL STQGCN VQC LAYER ABLATION EXPERIMENTS COMPLETED!")
print(f"Unified Metrics saved to: {ABLATION_CSV_PATH}")
print("=================================================================")
