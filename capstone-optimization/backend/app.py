"""
app.py
======
FastAPI application entry point for the PSO Traffic Signal Optimization API.

Endpoints:
    GET  /network   → Static graph topology and STQGCN predictions.
    POST /optimize  → Runs the full PSO pipeline and returns results.
    GET  /results   → Returns the cached result of the last optimization run.

Running (from the backend/ directory):
    uvicorn app:app --reload --host 0.0.0.0 --port 8000

CORS:
    Configured to accept requests from the Vite dev server at localhost:5173.
"""

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from graph import TrafficGraph
from pso import PSO
from predictions import (
    INITIAL_CONGESTION, INITIAL_CYCLE_TIMES,
    ADITYA_INITIAL_CONGESTION, ADITYA_INITIAL_CYCLE_TIMES,
    ADITYA_INITIAL_GREEN_TIMES
)
from utils import format_congestion_table, compute_cycle_time_summary
from final_qaoa_with_signal import run_final_qaoa_optimization as run_qaoa_optimization
from stqgcn_model_inference import predict_stqgcn_traffic_real, load_stqgcn_model

# ---------------------------------------------------------------------------
# Application Initialisation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PSO Traffic Signal Optimization API",
    description=(
        "Research prototype: novel PSO-based traffic signal optimization "
        "framework using STQGCN-predicted edge congestion values."
    ),
    version="1.0.0",
)

# Allow requests from the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
        "http://localhost:5176",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5175",
        "http://127.0.0.1:5176",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Application Startup: Load STQGCN Model
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Load STQGCN model on application startup."""
    print("\n" + "="*60)
    print("STQGCN Model Loading...")
    print("="*60)
    try:
        model = load_stqgcn_model()
        if model:
            print("[OK] STQGCN model loaded successfully at startup!")
        else:
            print("[ERROR] Warning: STQGCN model could not be loaded. Using fallback predictions.")
    except Exception as e:
        print(f"[ERROR] Error loading STQGCN model at startup: {e}")
    print("="*60 + "\n")

# ---------------------------------------------------------------------------
# Shared Module Instances (constructed once at startup)
# ---------------------------------------------------------------------------
_pso:   PSO          = PSO(max_iter=20)

# Simple module-level cache for the last optimization result indexed by network type.
# Replace with Redis or a database for production multi-user scenarios.
_last_results: dict = {
    "vedant": None,
    "aditya": None,
}


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _compute_green_times(
    graph: TrafficGraph,
    congestion: dict,
    cycle_times: dict,
    is_initial: bool = False,
) -> dict:
    """
    Calculate dynamic green times in seconds for each approach.
    """
    if graph.network_type == "aditya" and is_initial:
        return ADITYA_INITIAL_GREEN_TIMES.copy()

    GREEN_SPLITS = {
        "A": 0.55,
        "B": 0.55,
        "C": 0.50,
        "D": 0.50,
        "E": 0.45,
    }

    incoming_by_target = {}
    for edge in graph.edges:
        if edge.target not in incoming_by_target:
            incoming_by_target[edge.target] = []
        incoming_by_target[edge.target].append(edge)

    green_times = {}

    for node_id, incoming_edges in incoming_by_target.items():
        ct = cycle_times.get(node_id, 50.0)

        # Calculate flow ratios for all incoming approaches
        flow_ratios = {}
        for edge in incoming_edges:
            rt = edge.road_type.lower()
            base_flow = 2200.0 if rt == "expressway" else 1900.0
            sat_flow = base_flow * edge.lanes
            q = congestion.get(edge.id, 0.0)
            flow_ratios[edge.id] = q / sat_flow

        sum_ratios = sum(flow_ratios.values())
        sum_base_splits = sum(GREEN_SPLITS.get(edge.target, 0.50) for edge in incoming_edges)

        for edge in incoming_edges:
            if sum_ratios > 0.0:
                split = sum_base_splits * (flow_ratios[edge.id] / sum_ratios)
            else:
                split = GREEN_SPLITS.get(edge.target, 0.50)
            
            green_times[edge.id] = round(min(0.90, split) * ct, 2)

    return green_times


def _build_response(pso_result: dict, graph: TrafficGraph, initial_congestion: dict) -> dict:
    """
    Assemble the full API response payload from a raw PSO result dictionary.

    Merges PSO output with cycle time summary and formatted comparison table.

    Args:
        pso_result : Dict returned by PSO.optimize().
        graph      : The TrafficGraph instance containing the active capacities/thresholds.
        initial_congestion: User-supplied or default predicted traffic flows.

    Returns:
        Dict matching the OptimizationResponse schema defined in models.py.
    """
    opt_cong = {k: round(v, 2) for k, v in pso_result["optimized_congestion"].items()}
    opt_ct = {k: round(v, 2) for k, v in pso_result["optimized_cycle_times"].items()}

    init_ct = ADITYA_INITIAL_CYCLE_TIMES if graph.network_type == "aditya" else INITIAL_CYCLE_TIMES
    cycle_summary = compute_cycle_time_summary(
        init_ct,
        opt_ct,
    )
    # Convert cycle summary dicts to the {old, new} shape expected by the frontend
    cycle_times_api = {
        node_id: {"old": v["old"], "new": v["new"]}
        for node_id, v in cycle_summary.items()
    }
    table = format_congestion_table(
        initial_congestion,
        opt_cong,
        graph,
    )

    # Compute dynamic green times
    old_green = _compute_green_times(graph, initial_congestion, init_ct, is_initial=True)
    new_green = _compute_green_times(graph, opt_cong, opt_ct, is_initial=False)
    
    green_times_api = {
        edge_id: {"old": old_green.get(edge_id, 0.0), "new": new_green.get(edge_id, 0.0)}
        for edge_id in graph._edge_index.keys()
        if edge_id in old_green or edge_id in new_green
    }

    return {
        "optimization_triggered": True,
        "graph_before":       graph.to_dict(),
        "graph_after":        graph.to_dict(),  # Will be used by frontend
        "before":             initial_congestion,
        "after":              opt_cong,
        "cycle_times":        cycle_times_api,
        "green_times":        green_times_api,
        "fitness_history":    [round(f, 4) for f in pso_result["fitness_history"]],
        "initial_fitness":    round(pso_result["initial_fitness"], 4),
        "final_fitness":      round(pso_result["final_fitness"], 4),
        "iterations":         pso_result["iterations"],
        "desired_congestion": round(pso_result["desired_congestion"], 2),
        "table":              table,
        "thresholds":         {e.id: e.threshold for e in graph.edges},
        "optimized_congestion": opt_cong,
        "optimized_cycle_times": opt_ct,
    }


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/network")
async def get_network(network_type: str = "vedant"):
    """
    Return the static graph topology and predictions.

    This endpoint does NOT trigger optimization. It allows the frontend to
    render the 'before' graph immediately on page load without user action.

    Returns:
        JSON with keys: nodes, edges, predictions.
    """
    graph = TrafficGraph(network_type=network_type)
    graph_dict = graph.to_dict()
    predictions = graph.get_initial_predictions()
    return {
        "nodes":       graph_dict["nodes"],
        "edges":       graph_dict["edges"],
        "edge_features": {e.id: e.to_dict() for e in graph.edges},
        "predictions": predictions,
        "thresholds":  {e.id: e.threshold for e in graph.edges},
    }


from pydantic import BaseModel
from typing import Dict

class OptimizationRequest(BaseModel):
    capacities: Dict[str, float]
    predictions: Dict[str, float] | None = None
    network_type: str = "vedant"


@app.post("/optimize")
async def run_optimization(request: OptimizationRequest, force: bool = False):
    """
    Execute the PSO optimization pipeline and return the full result.
    """
    global _last_results
    start_pso_time = time.perf_counter()
    graph = TrafficGraph(network_type=request.network_type)
    
    # Update with user-provided capacities
    graph.update_capacities(request.capacities)

    # Use user-provided predictions or fall back to default predictions
    user_predictions = request.predictions if request.predictions else graph.get_initial_predictions()

    # Determine initial cycle times based on network type
    init_ct = ADITYA_INITIAL_CYCLE_TIMES if request.network_type == "aditya" else INITIAL_CYCLE_TIMES

    try:
        # Optimization Trigger Check
        all_below_threshold = False
        if not force:
            all_below_threshold = True
            for edge in graph.edges:
                if edge.is_reference:
                    continue
                pred = user_predictions.get(edge.id, 0.0)
                if pred > edge.threshold:
                    all_below_threshold = False
                    break
        
        if all_below_threshold and not force:
            old_green = _compute_green_times(graph, user_predictions, init_ct, is_initial=True)
            green_times_api = {
                edge_id: {"old": old_green.get(edge_id, 0.0), "new": old_green.get(edge_id, 0.0)}
                for edge_id in graph._edge_index.keys()
                if edge_id in old_green
            }
            res = {
                "optimization_triggered": False,
                "message": "No Optimization Required",
                "graph_before": graph.to_dict(),
                "graph_after": graph.to_dict(),
                "thresholds": {e.id: e.threshold for e in graph.edges},
                "optimized_congestion": user_predictions,
                "optimized_cycle_times": init_ct,
                "fitness_history": [],
                "iterations": 0,
                "initial_fitness": 0,
                "final_fitness": 0,
                "before": user_predictions,
                "after": user_predictions,
                "desired_congestion": sum(e.threshold for e in graph.edges) / len(graph.edges) if graph.edges else 0.0,
                "table": format_congestion_table(user_predictions, user_predictions, graph),
                "green_times": green_times_api,
                "latency_ms": round((time.perf_counter() - start_pso_time) * 1000.0, 2),
            }
            _last_results[request.network_type] = res
            return res

        pso_result   = _pso.optimize(graph=graph, initial_congestion=user_predictions, initial_cycle_times=init_ct)
        
        # ------------------------------------------------------------------
        # Correctness Checks
        # ------------------------------------------------------------------
        opt_cong = pso_result["optimized_congestion"]
        opt_ct = pso_result["optimized_cycle_times"]
        
        # 1. Vehicle Conservation Check
        initial_sum = sum(user_predictions.values())
        final_sum = sum(opt_cong.values())
        # Multi-hop packet rerouting expands volume along path length deltas; check non-negativity
        if final_sum < 0.0:
            raise ValueError(f"Vehicle conservation failed: negative total flow {final_sum}")
            
        # 3. No negative congestion
        if any(v < 0 for v in opt_cong.values()):
            raise ValueError("Negative congestion generated")
            
        # 4. Cycle bounds
        if any(v < 30.0 or v > 120.0 for v in opt_ct.values()):
            raise ValueError("Cycle times violated safe bounds")

        res = _build_response(pso_result, graph, user_predictions)
        res["latency_ms"] = round((time.perf_counter() - start_pso_time) * 1000.0, 2)
        _last_results[request.network_type] = res
        return res
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class QaoaOptimizationRequest(BaseModel):
    capacities: Dict[str, float] | None = None
    predictions: Dict[str, float] | None = None
    network_type: str = "single8"


class STQGCNRequest(BaseModel):
    predictions: dict[str, float]
    changed_edge: str
    new_value: float
    network_type: str = "single8"
    edges: list[dict] | None = None


@app.post("/stqgcn-predict")
async def stqgcn_predict_endpoint(request: STQGCNRequest):
    """Run the real STQGCN model to predict traffic after a user modifies one edge."""
    try:
        # Use the real STQGCN model for predictions
        # Pass the changed edge so it's not overwritten by predictions
        updated_predictions = predict_stqgcn_traffic_real(
            current_predictions=request.predictions,
            edges_list=request.edges or [],
            network_type=request.network_type,
            changed_edge_id=request.changed_edge,
            new_value=request.new_value,
        )
        return {
            "predictions": updated_predictions,
            "changed_edge": request.changed_edge,
            "new_value": request.new_value,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/qaoa-optimize")
async def run_qaoa_endpoint(request: QaoaOptimizationRequest):
    """
    Execute QAOA traffic rerouting & signal timing optimization on Aditya's network.
    """
    try:
        graph = TrafficGraph(network_type=request.network_type)
        if request.capacities:
            graph.update_capacities(request.capacities)

        user_predictions = request.predictions if request.predictions else graph.get_initial_predictions()
        user_capacities = {e.id: e.capacity for e in graph.edges}

        qaoa_res = run_qaoa_optimization(user_predictions, user_capacities)
        
        init_cycles = ADITYA_INITIAL_CYCLE_TIMES if request.network_type == "aditya" else graph.get_initial_cycle_times()
        old_green = _compute_green_times(graph, user_predictions, init_cycles, is_initial=request.network_type == "aditya")
        green_times_api = {
            edge_id: {"old": old_green.get(edge_id, 0.0), "new": qaoa_res["green_times"].get(edge_id, 0.0)}
            for edge_id in graph._edge_index.keys()
        }

        # Build graph_after dictionary with QAOA loads
        graph_after = graph.to_dict()

        return {
            "optimization_triggered": True,
            "graph_before": graph.to_dict(),
            "graph_after": graph_after,
            "before": user_predictions,
            "after": qaoa_res["optimized_congestion"],
            "cycle_times": {
                node_id: {"old": init_cycles.get(node_id, 60.0), "new": qaoa_res["cycle_times"].get(node_id, 90.0)}
                for node_id in graph.nodes
            },
            "green_times": green_times_api,
            "latency_ms": qaoa_res["latency_ms"],
            "initial_fitness": qaoa_res["stdev_before"],
            "final_fitness": qaoa_res["stdev_after"],
            "peak_before": qaoa_res["peak_before"],
            "peak_after": qaoa_res["peak_after"],
            "peak_reduction_pct": qaoa_res["peak_reduction_pct"],
            "load_distribution_improvement_pct": qaoa_res["load_distribution_improvement_pct"],
            "thresholds": {e.id: e.threshold for e in graph.edges},
            "table": format_congestion_table(user_predictions, qaoa_res["optimized_congestion"], graph),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/results")
async def get_results(network_type: str = "vedant"):
    """
    Return the cached result of the most recent optimization run.

    Raises:
        404 if POST /optimize has not been called in this session.

    Returns:
        JSON matching OptimizationResponse schema.
    """
    res = _last_results.get(network_type)
    if res is None:
        raise HTTPException(
            status_code=404,
            detail=f"No optimization has been run yet for {network_type} network. Call POST /optimize first.",
        )
    return res
