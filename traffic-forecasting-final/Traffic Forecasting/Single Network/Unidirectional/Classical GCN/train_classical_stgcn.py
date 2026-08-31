import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../Dataset/Single Network/mumbai_stqgcn_dataset_10k.npz"))
TOPOLOGY_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../Dataset/Single Network/graph_topology.json"))

# Output CSV paths
METRICS_CSV_PATH = os.path.join(SCRIPT_DIR, "classical_stgcn_performance_metrics.csv")
HISTORY_CSV_PATH = os.path.join(SCRIPT_DIR, "classical_stgcn_epoch_history.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "classical_stgcn_model.pt")

# --- Load Dataset ---
print(f"Loading dataset from: {DATASET_PATH}")
dataset = np.load(DATASET_PATH)

X_node = dataset["node_features"]       # (10000, 8, 6)
E_edge = dataset["edge_features"]       # (10000, 11, 6)
T_temp = dataset["temporal_features"]   # (10000, 4)
Y_target = dataset["target_flow"]       # (10000, 11)

num_timesteps, num_nodes, node_feat_dim = X_node.shape
_, num_edges, edge_feat_dim = E_edge.shape
_, temp_feat_dim = T_temp.shape

print(f"Dataset shape: Timesteps={num_timesteps}, Nodes={num_nodes}, Edges={num_edges}")

# Calculate Normalization Statistics for Denormalization (Eq 12)
mu_traffic = float(np.mean(Y_target))
std_traffic = float(np.std(Y_target))
print(f"Traffic Forecasting (5s Lookahead) Target Normalization Stats: mean = {mu_traffic:.2f}, std = {std_traffic:.2f}")

# Normalize Node and Edge features
X_mean, X_std = np.mean(X_node, axis=(0, 1)), np.std(X_node, axis=(0, 1)) + 1e-6
E_mean, E_std = np.mean(E_edge, axis=(0, 1)), np.std(E_edge, axis=(0, 1)) + 1e-6

X_node_norm = (X_node - X_mean) / X_std
E_edge_norm = (E_edge - E_mean) / E_std
Y_target_norm = (Y_target - mu_traffic) / std_traffic

# Directed Graph Edges (Unidirectional Topology)
# 1. V1->V2, V4, V5
# 2. V2->V3, V5
# 3. V3->V8
# 4. V4->V5
# 5. V5->V6
# 6. V6->V2, V7
# 7. V7->V3
EDGES = [
    (0, 1), (0, 3), (0, 4), # V1 -> V2, V4, V5
    (1, 2), (1, 4),         # V2 -> V3, V5
    (2, 7),                 # V3 -> V8
    (3, 4),                 # V4 -> V5
    (4, 5),                 # V5 -> V6
    (5, 1), (5, 6),         # V6 -> V2, V7
    (6, 2)                  # V7 -> V3
]

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
Y_raw_test = Y_raw_targets[train_size+val_size:] # Raw physical traffic volume for evaluation

# PyTorch DataLoaders
batch_size = 64
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(E_train), torch.FloatTensor(T_train), torch.FloatTensor(Y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(E_val), torch.FloatTensor(T_val), torch.FloatTensor(Y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(E_test), torch.FloatTensor(T_test), torch.FloatTensor(Y_raw_test))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Classical STGCN Model Definition (Math Model - Version 2) ---
class ClassicalSTGCN(nn.Module):
    def __init__(self, num_nodes, num_edges, node_in_dim, edge_in_dim, temp_in_dim, hidden_dim=16):
        super(ClassicalSTGCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.edges = EDGES
        self.hidden_dim = hidden_dim
        
        # 1. Contextualized Node Embedding (Eq 2)
        # Input: [x_i,t || w_i,t || tau_t] -> dimension = node_in_dim + temp_in_dim
        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim + temp_in_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2. Spatio-Temporal Message Passing (Eq 3-5)
        # Input: [h_j,t || e_ji,t || tau_t] -> dimension = hidden_dim + edge_in_dim + temp_in_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_in_dim + temp_in_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 3. Classical Replacement for Quantum Layer (Eq 7-14)
        # Classical feature enhancement layer replacing QVC
        self.classical_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # 4. Edge Prediction Head (Eq 15)
        # Input: [h_src || h_dst || e_ij || tau_t] -> dimension = 2*hidden_dim + edge_in_dim + temp_in_dim
        self.pred_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_in_dim + temp_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, X_node, E_edge, T_temp):
        B = X_node.size(0)
        
        # --- Step 1: Contextualized Node Embedding (Eq 2) ---
        # T_temp expanded to all nodes: (B, num_nodes, temp_in_dim)
        T_expanded = T_temp.unsqueeze(1).expand(-1, self.num_nodes, -1)
        node_inputs = torch.cat([X_node, T_expanded], dim=-1) # (B, num_nodes, node_in_dim + temp_in_dim)
        h_0 = self.node_embed(node_inputs) # (B, num_nodes, hidden_dim)
        
        # --- Step 2: Spatio-Temporal Message Passing & Aggregation (Eq 3-5) ---
        a_aggregated = torch.zeros_like(h_0)
        
        for e_idx, (src, dst) in enumerate(self.edges):
            h_src = h_0[:, src, :] # (B, hidden_dim)
            e_feat = E_edge[:, e_idx, :] # (B, edge_in_dim)
            t_feat = T_temp # (B, temp_in_dim)
            
            msg_input = torch.cat([h_src, e_feat, t_feat], dim=-1)
            msg = self.message_mlp(msg_input) # (B, hidden_dim)
            
            # Aggregate incoming messages (Sum aggregation)
            a_aggregated[:, dst, :] += msg
            
        # --- Step 3: Residual Update (Eq 6) ---
        h_1 = h_0 + a_aggregated
        
        # --- Step 4: Classical Layer Update (Eq 7-14) ---
        # Classical feature enhancement replacing Quantum Variational Circuit
        q_classical = self.classical_update(h_1)
        h_final = self.layer_norm(h_1 + q_classical) # (B, num_nodes, hidden_dim)
        
        # --- Step 5: Edge Prediction Head (Eq 15) ---
        predictions = []
        for e_idx, (src, dst) in enumerate(self.edges):
            h_s = h_final[:, src, :]
            h_d = h_final[:, dst, :]
            e_feat = E_edge[:, e_idx, :]
            t_feat = T_temp
            
            pred_input = torch.cat([h_s, h_d, e_feat, t_feat], dim=-1)
            pred_out = self.pred_mlp(pred_input).squeeze(-1) # (B,)
            predictions.append(pred_out)
            
        y_hat_norm = torch.stack(predictions, dim=1) # (B, num_edges)
        return y_hat_norm

# --- Initialize Model, Loss, and Optimizer ---
model = ClassicalSTGCN(
    num_nodes=num_nodes,
    num_edges=num_edges,
    node_in_dim=node_feat_dim,
    edge_in_dim=edge_feat_dim,
    temp_in_dim=temp_feat_dim,
    hidden_dim=16
).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# --- Training Loop ---
# Save initial model state checkpoint
torch.save(model.state_dict(), MODEL_SAVE_PATH)

NUM_EPOCHS = 40
print(f"Starting Training for {NUM_EPOCHS} Epochs...")

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
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS}] - Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} | Time: {epoch_duration:.2f}s")

total_training_time = time.time() - start_total_time
print(f"Training Complete in {total_training_time:.2f} seconds.")

# --- Save Epoch History CSV ---
history_df = pd.DataFrame(history)
history_df.to_csv(HISTORY_CSV_PATH, index=False)
print(f"Epoch history saved to: {HISTORY_CSV_PATH}")

# --- Test Evaluation & Metric Computation ---
print("\n--- Evaluating Model on Test Set ---")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

y_preds_raw_list = []
y_true_raw_list = []

with torch.no_grad():
    for batch_X, batch_E, batch_T, batch_Y_raw in test_loader:
        batch_X, batch_E, batch_T = batch_X.to(device), batch_E.to(device), batch_T.to(device)
        preds_norm = model(batch_X, batch_E, batch_T).cpu().numpy()
        
        # Denormalize predictions (Eq 12 from N=4 PDF)
        preds_raw = preds_norm * std_traffic + mu_traffic
        
        y_preds_raw_list.append(preds_raw)
        y_true_raw_list.append(batch_Y_raw.numpy())

y_pred_all = np.vstack(y_preds_raw_list) # (1500, 11)
y_true_all = np.vstack(y_true_raw_list) # (1500, 11)

# Calculate Evaluation Metrics
mae = float(np.mean(np.abs(y_true_all - y_pred_all)))
rmse = float(np.sqrt(np.mean((y_true_all - y_pred_all) ** 2)))
mape = float(np.mean(np.abs((y_true_all - y_pred_all) / (y_true_all + 1e-5))) * 100.0)

# R^2 Score
ss_res = np.sum((y_true_all - y_pred_all) ** 2)
ss_tot = np.sum((y_true_all - np.mean(y_true_all)) ** 2)
r2_score = float(1.0 - (ss_res / ss_tot))

print("\n================ FINAL TEST PERFORMANCE METRICS ================")
print(f"Model Architecture : Classical STGCN (Unidirectional Single Network)")
print(f"MAE (Veh/hr)       : {mae:.2f}")
print(f"RMSE (Veh/hr)      : {rmse:.2f}")
print(f"MAPE (%)           : {mape:.2f}%")
print(f"R2 Score           : {r2_score:.4f}")
print(f"Total Train Time   : {total_training_time:.2f} seconds")
print("=================================================================\n")

# Save Summary Performance Metrics CSV
metrics_summary = [{
    "model_name": "Forecast_5s_Classical_STGCN_Unidirectional_Single_Network",
    "network_type": "Single_Network_Unidirectional",
    "scale": "Junction_Level_N8",
    "quantum_enabled": False,
    "num_qubits": 0,
    "mae_veh_hr": round(mae, 4),
    "rmse_veh_hr": round(rmse, 4),
    "mape_percent": round(mape, 4),
    "r2_score": round(r2_score, 4),
    "best_val_mse": round(best_val_loss, 6),
    "total_train_time_sec": round(total_training_time, 2),
    "avg_epoch_time_sec": round(total_training_time / NUM_EPOCHS, 4)
}]

metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv(METRICS_CSV_PATH, index=False)
print(f"Performance metrics saved to: {METRICS_CSV_PATH}")
