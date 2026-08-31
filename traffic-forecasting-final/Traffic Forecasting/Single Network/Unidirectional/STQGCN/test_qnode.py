import pennylane as qml
import torch

NUM_QUBITS = 8
NUM_LAYERS = 2

dev = qml.device("default.qubit", wires=NUM_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def vqc_circuit(inputs, weights):
    for q in range(NUM_QUBITS):
        qml.RY(inputs[q], wires=q)
    for k in range(NUM_LAYERS):
        for q in range(NUM_QUBITS):
            qml.RZ(weights[k, q, 0], wires=q)
            qml.RY(weights[k, q, 1], wires=q)
            qml.RZ(weights[k, q, 2], wires=q)
        for q in range(NUM_QUBITS):
            qml.CNOT(wires=[q, (q + 1) % NUM_QUBITS])
    return [qml.expval(qml.PauliZ(q)) for q in range(NUM_QUBITS)]

class QuantumLayerModule(torch.nn.Module):
    def __init__(self, n_layers, n_qubits=8):
        super().__init__()
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.weights = torch.nn.Parameter(torch.randn(n_layers, n_qubits, 3) * 0.1)

    def forward(self, x):
        # x is (Batch, 8)
        # Vectorized QNode call over batch
        res = [torch.stack(vqc_circuit(x[i], self.weights), dim=0) for i in range(x.shape[0])]
        return torch.stack(res, dim=0)

module = QuantumLayerModule(n_layers=2, n_qubits=8)
x = torch.randn(16, 8)
out = module(x)
print("Output shape:", out.shape)
