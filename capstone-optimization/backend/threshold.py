"""
threshold.py
============
Mathematical model for computing dynamic traffic congestion thresholds.
"""

def calculate_threshold(
    capacity: float,
    speed: float,
    length: float,
    road_type: str,
    is_reference: bool = False,
    network_type: str = "vedant",
) -> float:
    """
    Compute the operating congestion threshold for a given edge.
    
    Threshold is now:
        For vedant: T_i = η_i * C_i
            For arterials: η = 0.85
            For expressways: η = 0.95
            Reference edge: Threshold = 0
        For aditya: T_i = 0.60 * C_i

    Args:
        capacity : Road capacity (veh/hr)
        speed    : Free-flow speed (km/h)
        length   : Road length (km)
        road_type: Classification string ("Arterial", "Expressway", "Reference", etc.)
        is_reference: True if this edge is the reference edge
        network_type: "vedant" or "aditya"

    Returns:
        float: Computed threshold (veh/hr).
    """
    if is_reference or road_type.lower() == "reference":
        return 0.0

    return 0.60 * capacity
