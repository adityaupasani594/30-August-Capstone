import os
import time
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
print(f"Using PyTorch device: {device}")

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../Dataset/Single Network/mumbai_stqgcn_dataset_10k.npz"))

METRICS_CSV_PATH = os.path.join(SCRIPT_DIR, "single_bidirectional_lstm_performance_metrics.csv")
HISTORY_CSV_PATH = os.path.join(SCRIPT_DIR, "single_bidirectional_lstm_epoch_history.csv")
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "single_bidirectional_lstm_model.pt")

# --- Load Single Master Dataset ---
print(f"Loading master dataset from: {DATASET_PATH}")
dataset = np.load(DATASET_PATH)

X_node = dataset["node_features"]       # (10000, 8, 6)
E_edge_raw = dataset["edge_features"]   # (10000, 11, 6)
T_temp = dataset["temporal_features"]   # (10000, 4)
Y_target_raw = dataset["target_flow"]   # (10000, 11)

num_timesteps, num_nodes, node_feat_dim = X_node.shape
_, num_forward_edges, edge_feat_dim = E_edge_raw.shape
_, temp_feat_dim = T_temp.shape

# --- Construct Bidirectional Topology (22 Edges) In Memory ---
FORWARD_EDGES = [
    (0, 1), (0, 3), (0, 4),
    (1, 2), (1, 4),
    (2, 7),
    (3, 4),
    (4, 5),
    (5, 1), (5, 6),
    (6, 2)
]
REVERSE_EDGES = [(dst, src) for src, dst in FORWARD_EDGES]
EDGES_BI = FORWARD_EDGES + REVERSE_EDGES
num_edges_bi = len(EDGES_BI)

print(f"Bidirectional Single Network Topology: {num_nodes} Nodes, {num_edges_bi} Directed Edges (11F + 11R)")

# Build 22-Edge Feature & Target Tensors
E_edge_reverse = E_edge_raw.copy()
E_edge_reverse[:, :, 0] *= 0.95
E_edge_bi = np.concatenate([E_edge_raw, E_edge_reverse], axis=1)  # (10000, 22, 6)

Y_target_reverse = Y_target_raw * (0.8 + 0.3 * np.sin(np.linspace(0, 100, num_timesteps))[:, None])
Y_target_bi = np.concatenate([Y_target_raw, Y_target_reverse], axis=1)  # (10000, 22)

# Normalization
mu_traffic = float(np.mean(Y_target_bi))
std_traffic = float(np.std(Y_target_bi))
print(f"Bidirectional Target Flow Normalization Stats: mean = {mu_traffic:.2f}, std = {std_traffic:.2f}")

X_mean, X_std = np.mean(X_node, axis=(0, 1)), np.std(X_node, axis=(0, 1)) + 1e-6
E_mean, E_std = np.mean(E_edge_bi, axis=(0, 1)), np.std(E_edge_bi, axis=(0, 1)) + 1e-6

X_node_norm = (X_node - X_mean) / X_std
E_edge_norm = (E_edge_bi - E_mean) / E_std
Y_target_norm = (Y_target_bi - mu_traffic) / std_traffic

# --- Flatten features for LSTM input ---
# Input: [node_features_flat | edge_features_flat | temporal_features]
# Shape: (10000, 8*6 + 22*6 + 4) = (10000, 184)
X_node_flat = X_node_norm.reshape(num_timesteps, -1)       # (10000, 48)
E_edge_flat = E_edge_norm.reshape(num_timesteps, -1)       # (10000, 132)
input_features = np.concatenate([X_node_flat, E_edge_flat, T_temp], axis=1)  # (10000, 184)
input_dim = input_features.shape[1]

print(f"LSTM Input Dimension: {input_dim} (node:{num_nodes*node_feat_dim} + edge:{num_edges_bi*edge_feat_dim} + temp:{temp_feat_dim})")

# --- 5-Second Lookahead Forecasting Target Shift (Features at t -> Target at t+1) ---
X_features = input_features[:-1]
Y_targets = Y_target_norm[1:]
Y_raw_targets = Y_raw_test = (Y_target_bi[1:] if 'Y_target_bi' in locals() else Y_target_raw[1:])

num_samples = len(X_features)
train_size = int(0.70 * num_samples)
val_size = int(0.15 * num_samples)
test_size = num_samples - train_size - val_size

X_train = X_features[:train_size]
X_val = X_features[train_size:train_size+val_size]
X_test = X_features[train_size+val_size:]

Y_train = Y_targets[:train_size]
Y_val = Y_targets[train_size:train_size+val_size]
Y_raw_test = Y_raw_targets[train_size+val_size:]
Y_raw_test = Y_raw_targets[train_size+val_size:]

batch_size = 64
train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(Y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(Y_raw_test))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- LSTM Model ---
class TrafficLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)          # (batch, 1, input_dim)
        lstm_out, _ = self.lstm(x)  # (batch, 1, hidden_dim)
        return self.head(lstm_out[:, -1, :])

# Initialize Model
HIDDEN_DIM = 128
NUM_LAYERS = 2

model = TrafficLSTM(
    input_dim=input_dim,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    output_dim=num_edges_bi,
    dropout=0.2
).to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nLSTM Model: hidden_dim={HIDDEN_DIM}, num_layers={NUM_LAYERS}, output_dim={num_edges_bi}")
print(f"Total Trainable Parameters: {total_params:,}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# --- Training Loop ---
NUM_EPOCHS = 40
print(f"\nStarting Bidirectional Single Network LSTM Training for {NUM_EPOCHS} Epochs...")

history = []
start_total_time = time.time()
best_val_loss = float("inf")

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    model.train()
    train_loss = 0.0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        preds = model(batch_X)
        loss = criterion(preds, batch_Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    train_loss /= train_size

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            preds = model(batch_X)
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
print(f"\nTraining Complete in {total_training_time:.2f} seconds.")

pd.DataFrame(history).to_csv(HISTORY_CSV_PATH, index=False)
print(f"Epoch history saved to: {HISTORY_CSV_PATH}")

# --- Test Evaluation ---
print("\n--- Evaluating LSTM on Test Set ---")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

y_preds_raw_list, y_true_raw_list = [], []
with torch.no_grad():
    for batch_X, batch_Y_raw in test_loader:
        batch_X = batch_X.to(device)
        preds_norm = model(batch_X).cpu().numpy()
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

print("\n======== BIDIRECTIONAL SINGLE NETWORK LSTM TEST METRICS ========")
print(f"Model Architecture    : 2-Layer LSTM (Bidirectional Single Network)")
print(f"Network Topology      : 8 Nodes, 22 Directed Edges (11F + 11R)")
print(f"LSTM Hidden Dim       : {HIDDEN_DIM} | Layers: {NUM_LAYERS} | Params: {total_params:,}")
print(f"MAE  (Veh/hr)         : {mae:.2f}")
print(f"RMSE (Veh/hr)         : {rmse:.2f}")
print(f"MAPE (%)              : {mape:.2f}%")
print(f"R2 Score              : {r2_score:.4f}")
print(f"Total Train Time      : {total_training_time:.2f} seconds")
print("=================================================================\n")

metrics_summary = [{
    "model_name": "Forecast_5s_LSTM_Bidirectional_Single_8N",
    "network_type": "Single_Network_Bidirectional",
    "scale": "8_Nodes_22_Edges",
    "model_type": "LSTM",
    "hidden_dim": HIDDEN_DIM,
    "num_layers": NUM_LAYERS,
    "total_params": total_params,
    "quantum_enabled": False,
    "mae_veh_hr": round(mae, 4),
    "rmse_veh_hr": round(rmse, 4),
    "mape_percent": round(mape, 4),
    "r2_score": round(r2_score, 4),
    "best_val_mse": round(best_val_loss, 6),
    "total_train_time_sec": round(total_training_time, 2),
    "avg_epoch_time_sec": round(total_training_time / NUM_EPOCHS, 4)
}]

pd.DataFrame(metrics_summary).to_csv(METRICS_CSV_PATH, index=False)
print(f"Performance metrics saved to: {METRICS_CSV_PATH}")
