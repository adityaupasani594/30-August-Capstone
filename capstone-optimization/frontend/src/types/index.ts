export interface NodeData {
  id: string;
  label: string;
  initial_cycle_time: number;
}

export interface EdgeData {
  id: string;
  source: string;
  target: string;
  weight: number;
  capacity: number;
  speed: number;
  lanes: number;
  length: number;
  road_type: string;
  threshold: number;
  is_reference: boolean;
}

export interface NetworkResponse {
  nodes: NodeData[];
  edges: EdgeData[];
  edge_features: Record<string, EdgeData>;
  predictions: Record<string, number>;
  thresholds: Record<string, number>;
}

export interface OptimizationRequest {
  capacities: Record<string, number>;
  predictions?: Record<string, number>;
}

export interface CycleTimeChange {
  old: number;
  new: number;
}

export interface OptimizationResponse {
  optimization_triggered: boolean;
  message?: string;
  graph_before: { nodes: NodeData[]; edges: EdgeData[] };
  graph_after: { nodes: NodeData[]; edges: EdgeData[] };
  before?: Record<string, number>;
  after?: Record<string, number>;
  cycle_times?: Record<string, CycleTimeChange>;
  green_times?: Record<string, { old: number; new: number }>;
  fitness_history?: number[];
  initial_fitness?: number;
  final_fitness?: number;
  iterations?: number;
  desired_congestion?: number;
  table?: any;
  thresholds: Record<string, number>;
  optimized_congestion: Record<string, number>;
  optimized_cycle_times: Record<string, number>;
  latency_ms?: number;
  peak_before?: number;
  peak_after?: number;
  peak_reduction_pct?: number;
  load_distribution_improvement_pct?: number;
}

export interface ScenarioResult {
  scenarioName: string;
  initialFitness: number;
  finalFitness: number;
  fitnessImprovement: number;
  congestedEdges: number;
  fitnessHistory: number[];
}
