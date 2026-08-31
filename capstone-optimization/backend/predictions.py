"""
predictions.py
==============
Hardcoded STQGCN (Spatial-Temporal Quantum Graph Convolutional Network) predictions
for the 5-node demonstration traffic network.

In a production system, this module is the ONLY file that needs to change to
support live model inference. All downstream modules consume these constants.

Research Context:
    These predictions serve as the INPUT to the PSO-based signal optimization
    framework. The PSO module uses these values to compute the desired balanced
    distribution Q* and optimizes signal timings to minimize congestion imbalance.
"""

# ---------------------------------------------------------------------------
# STQGCN Predicted Edge Congestion (veh/hr)
# ---------------------------------------------------------------------------
# Replace this dictionary with live STQGCN model output in production.
INITIAL_CONGESTION: dict[str, float] = {
    "A\u2192B": 1288.52,
    "B\u2192A": 1200.00,
    "A\u2192C": 1319.37,
    "C\u2192A": 1250.00,
    "A\u2192D": 1333.36,
    "D\u2192A": 1300.00,
    "B\u2192D": 1386.11,
    "D\u2192B": 1150.00,
    "C\u2192D": 1361.65,
    "D\u2192C": 1100.00,
    "D\u2192E": 1285.58,
    "E\u2192D": 1250.00,
    "E\u2192B": 1350.00,   # Treated as normal arterial edge
    "B\u2192E": 1300.00,
}

# ---------------------------------------------------------------------------
# Edge Importance Weights
# ---------------------------------------------------------------------------
# Weights represent the relative importance of each edge in the objective
# function J = Σ w_i * (q_i - q_i*)².
#
#   Internal arterial edges : weight = 1  (moderate significance)
#   Incoming / outgoing edges: weight = 2  (boundary flows — higher priority)
#   Reference edge           : weight = 0  (excluded from optimization)
#
# These weights are FIXED and must NOT be editable from the frontend or API.
# ---------------------------------------------------------------------------
EDGE_WEIGHTS: dict[str, float] = {
    "A\u2192B": 1.0,   # Internal arterial
    "B\u2192A": 1.0,
    "A\u2192C": 1.0,   # Internal arterial
    "C\u2192A": 1.0,
    "A\u2192D": 2.0,   # Incoming to network core (higher weight)
    "D\u2192A": 2.0,
    "B\u2192D": 1.0,   # Internal arterial
    "D\u2192B": 1.0,
    "C\u2192D": 1.0,   # Internal arterial
    "D\u2192C": 1.0,
    "D\u2192E": 2.0,   # Outgoing from network core (higher weight)
    "E\u2192D": 2.0,
    "E\u2192B": 1.0,   # Treated as normal arterial edge
    "B\u2192E": 1.0,
}

# ---------------------------------------------------------------------------
# Static Edge Features
# ---------------------------------------------------------------------------
# Hardcoded edge features containing [capacity, speed, lanes, length, road_type]
# ---------------------------------------------------------------------------
EDGE_FEATURES: dict[str, dict] = {
    "A\u2192B": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 1.20, "road_type": "Arterial"},
    "B\u2192A": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 1.20, "road_type": "Arterial"},
    "A\u2192C": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 0.95, "road_type": "Arterial"},
    "C\u2192A": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 0.95, "road_type": "Arterial"},
    "A\u2192D": {"capacity": 2200.0, "speed": 60.0, "lanes": 4, "length": 1.60, "road_type": "Expressway"},
    "D\u2192A": {"capacity": 2200.0, "speed": 60.0, "lanes": 4, "length": 1.60, "road_type": "Expressway"},
    "B\u2192D": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 1.10, "road_type": "Arterial"},
    "D\u2192B": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 1.10, "road_type": "Arterial"},
    "C\u2192D": {"capacity": 1800.0, "speed": 40.0, "lanes": 2, "length": 0.80, "road_type": "Arterial"},
    "D\u2192C": {"capacity": 1800.0, "speed": 40.0, "lanes": 2, "length": 0.80, "road_type": "Arterial"},
    "D\u2192E": {"capacity": 2200.0, "speed": 60.0, "lanes": 4, "length": 1.40, "road_type": "Expressway"},
    "E\u2192D": {"capacity": 2200.0, "speed": 60.0, "lanes": 4, "length": 1.40, "road_type": "Expressway"},
    "E\u2192B": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 1.00, "road_type": "Arterial"},
    "B\u2192E": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 1.00, "road_type": "Arterial"},
}

# ---------------------------------------------------------------------------
# Initial Signal Cycle Times (seconds)
# ---------------------------------------------------------------------------
# Each node has a single traffic signal controller.
# These are the baseline values BEFORE PSO optimization.
# ---------------------------------------------------------------------------
INITIAL_CYCLE_TIMES: dict[str, float] = {
    "A": 40.0,
    "B": 60.0,
    "C": 45.0,
    "D": 70.0,
    "E": 50.0,
}

# ---------------------------------------------------------------------------
# Fixed Algorithm Constants
# ---------------------------------------------------------------------------

# # Identifier for the reference edge (excluded from all optimization steps).
# # This edge is used only as a baseline and is fixed at 0 veh/hr.
REFERENCE_EDGE: str = ""

# Cycle time safety bounds (seconds) applied during PSO and signal adjustment.
CYCLE_TIME_MIN: float = 30.0
CYCLE_TIME_MAX: float = 120.0

# ---------------------------------------------------------------------------
# Aditya's Network Configurations
# ---------------------------------------------------------------------------

ADITYA_INITIAL_CONGESTION: dict[str, float] = {
    "A\u2192B": 1300.0,
    "B\u2192A": 1300.0,
    "A\u2192C": 720.0,
    "C\u2192A": 720.0,
    "A\u2192D": 595.0,
    "D\u2192A": 595.0,
    "B\u2192C": 1360.0,
    "C\u2192B": 1360.0,
    "B\u2192D": 510.0,
    "D\u2192B": 510.0,
    "C\u2192D": 630.0,
    "D\u2192C": 630.0,
}

ADITYA_EDGE_WEIGHTS: dict[str, float] = {
    "A\u2192B": 1.0,
    "B\u2192A": 1.0,
    "A\u2192C": 1.0,
    "C\u2192A": 1.0,
    "A\u2192D": 1.0,
    "D\u2192A": 1.0,
    "B\u2192C": 1.0,
    "C\u2192B": 1.0,
    "B\u2192D": 1.0,
    "D\u2192B": 1.0,
    "C\u2192D": 1.0,
    "D\u2192C": 1.0,
}

ADITYA_EDGE_FEATURES: dict[str, dict] = {
    "A\u2192B": {"capacity": 2000.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "B\u2192A": {"capacity": 2000.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "A\u2192C": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "C\u2192A": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "A\u2192D": {"capacity": 1700.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "D\u2192A": {"capacity": 1700.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "B\u2192C": {"capacity": 2000.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "C\u2192B": {"capacity": 2000.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "B\u2192D": {"capacity": 1700.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "D\u2192B": {"capacity": 1700.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "C\u2192D": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
    "D\u2192C": {"capacity": 1800.0, "speed": 50.0, "lanes": 3, "length": 1.0, "road_type": "Arterial"},
}

ADITYA_INITIAL_CYCLE_TIMES: dict[str, float] = {
    "A": 75.0,
    "B": 85.0,
    "C": 65.0,
    "D": 95.0,
}

ADITYA_INITIAL_GREEN_TIMES: dict[str, float] = {
    "A\u2192B": 38.0,
    "A\u2192C": 18.0,
    "A\u2192D": 30.0,
    "B\u2192A": 32.5,
    "B\u2192C": 31.0,
    "B\u2192D": 27.0,
    "C\u2192A": 20.0,
    "C\u2192B": 28.5,
    "C\u2192D": 34.0,
    "D\u2192A": 18.5,
    "D\u2192B": 12.0,
    "D\u2192C": 14.5,
}
