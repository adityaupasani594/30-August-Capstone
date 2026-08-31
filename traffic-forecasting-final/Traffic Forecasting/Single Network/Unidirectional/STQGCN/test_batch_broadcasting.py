import pennylane as qml
import torch
import time

NUM_QUBITS = 8
NUM_LAYERS = 2
BATCH_SIZE = 64

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def vqc_circuit_broadcast(inputs, weights):
    # inputs shape: (B, 8)
    for q in range(NUM_QUBITS):
        qml.RY(inputs[:, q], wires=q)
    for k in range(NUM_LAYERS):
        for q in range(NUM_QUBITS):
            qml.RZ(weights[k, q, 0], wires=q)
            qml.RY(weights[k, q, 1], wires=q)
            qml.RZ(weights[k, q, 2], wires=q)
        for q in range(NUM_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % NUM_QUBITS])
    return [qml.expval(qml.PauliZ(q)) for q in range(NUM_QUBITS)]

weights = torch.randn(NUM_LAYERS, NUM_QUBITS, 3)
x = torch.randn(BATCH_SIZE, NUM_QUBITS)

start = time.time()
out = vqc_circuit_broadcast(x, weights)
out_tensor = torch.stack(out, dim=1) # (B, 8)
elapsed = time.time() - start

print(f"Broadcast Batch Output shape: {out_tensor.shape}, Time: {elapsed:.4f}s")
