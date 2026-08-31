import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { NetworkResponse, OptimizationResponse, ScenarioResult } from '../types';
import { NetworkGraph } from './NetworkGraph';
import { CongestionTable } from './CongestionTable';
import { OptimizationSummary } from './OptimizationSummary';
import { CapacityConfig } from './CapacityConfig';
import { calculateThreshold } from '../utils/threshold';
import { ResearchMetricsTable } from './ResearchMetricsTable';
import { ResearchAnalytics } from './ResearchAnalytics';

interface DashboardProps {
  networkType: 'vedant' | 'aditya';
  onBack: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ networkType, onBack }) => {
  const [network, setNetwork] = useState<NetworkResponse | null>(null);
  const [result, setResult] = useState<OptimizationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [capacities, setCapacities] = useState<Record<string, number>>({});
  const [predictions, setPredictions] = useState<Record<string, number>>({});
  const [scenarios, setScenarios] = useState<ScenarioResult[]>([]);
  const [scenarioName, setScenarioName] = useState<string>('Morning Peak');
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  const runStqgcnLikePrediction = (
    basePredictions: Record<string, number>,
    changedEdgeId: string,
    newValue: number,
    edges: { id: string; source: string; target: string; capacity: number }[]
  ): Record<string, number> => {
    const next = { ...basePredictions };
    const currentValue = Number(basePredictions[changedEdgeId] ?? 0);
    const delta = Math.max(0, Number(newValue)) - currentValue;
    if (edges.length === 0) {
      next[changedEdgeId] = Math.max(0, Number(newValue));
      return next;
    }

    const editedEdge = edges.find((edge) => edge.id === changedEdgeId);
    if (!editedEdge) {
      next[changedEdgeId] = Math.max(0, Number(newValue));
      return next;
    }

    const affected = edges.filter((edge) => {
      if (edge.id === changedEdgeId) return false;
      return edge.source === editedEdge.source || edge.target === editedEdge.target || edge.source === editedEdge.target || edge.target === editedEdge.source;
    });

    const totalNearby = affected.reduce((sum, edge) => sum + Math.max(0, next[edge.id] ?? 0), 0);
    const totalDrift = Math.abs(delta) || 1;

    next[changedEdgeId] = Math.max(0, Number(newValue));

    affected.forEach((edge) => {
      const baseValue = Number(next[edge.id] ?? 0);
      const weight = totalNearby > 0 ? (baseValue / totalNearby) : 1 / Math.max(1, affected.length);
      const adjustment = delta * 0.65 * weight;
      const adjusted = baseValue + adjustment;
      const capFactor = edge.capacity > 0 ? Math.min(edge.capacity * 1.35, 2200) : 2200;
      next[edge.id] = Math.max(0, Math.min(adjusted, capFactor));
    });

    edges.forEach((edge) => {
      if (edge.id === changedEdgeId || affected.some((item) => item.id === edge.id)) return;
      const baseValue = Number(next[edge.id] ?? 0);
      const drift = totalDrift * 0.08 * (baseValue / Math.max(1, Object.values(next).reduce((sum, val) => sum + val, 0)));
      next[edge.id] = Math.max(0, Math.min(baseValue + drift, edge.capacity * 1.25 || 2200));
    });

    return next;
  };

  const handleClearHistory = () => {
    setScenarios([]);
  };

  useEffect(() => {
    fetchNetwork();
  }, [networkType]);

  const fetchNetwork = async () => {
    try {
      setLoading(true);
      const data = await api.getNetwork(networkType);
      setNetwork(data);
      
      // Initialize capacities state
      const initialCapacities: Record<string, number> = {};
      data.edges.forEach(edge => {
        initialCapacities[edge.id] = edge.capacity;
      });
      setCapacities(initialCapacities);

      // Initialize predictions state
      setPredictions(data.predictions);
    } catch (err: any) {
      setError(err.message || 'Failed to load network data.');
    } finally {
      setLoading(false);
    }
  };

  const handleCapacityChange = (edgeId: string, newCapacity: number) => {
    setCapacities(prev => ({ ...prev, [edgeId]: newCapacity }));
  };

  const handlePredictionChange = (edgeId: string, newPrediction: number) => {
    setPredictions((prev) => {
      if (!network) return prev;
      return runStqgcnLikePrediction(
        prev,
        edgeId,
        newPrediction,
        network.edges,
      );
    });
  };

  const handleOptimize = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Validate positive capacities
      for (const [edgeId, cap] of Object.entries(capacities)) {
        const edge = network?.edges.find(e => e.id === edgeId);
        if (edge && cap <= 0) {
          throw new Error(`Capacity for edge ${edgeId} must be positive.`);
        }
      }

      // Validate non-negative predictions
      for (const [edgeId, pred] of Object.entries(predictions)) {
        if (pred < 0) {
          throw new Error(`Predicted traffic for edge ${edgeId} cannot be negative.`);
        }
      }

      const data = await api.runOptimization({ capacities, predictions, network_type: networkType });
      setResult(data);

      // Compute thresholds for analytics
      const localThresholds: Record<string, number> = {};
      network?.edges.forEach(edge => {
        localThresholds[edge.id] = calculateThreshold(
          capacities[edge.id] ?? edge.capacity,
          edge.speed,
          edge.length,
          edge.road_type,
          edge.is_reference,
          networkType
        );
      });

      // Count congested edges
      const congestedCount = network?.edges.filter(e => {
        const thresh = localThresholds[e.id] ?? e.threshold;
        const pred = predictions[e.id] ?? 0;
        return pred > thresh;
      }).length ?? 0;

      // Extract and fallback fitness values if optimization skipped
      let initialFitness = data.initial_fitness ?? 0;
      let finalFitness = data.final_fitness ?? 0;
      let fitnessHistory = data.fitness_history ?? [];

      if (!data.optimization_triggered) {
        let sumFitness = 0;
        network?.edges.forEach(e => {
          if (!e.is_reference) {
            const q = predictions[e.id] ?? 0;
            const t = localThresholds[e.id] ?? e.threshold;
            if (t > 0) {
              sumFitness += Math.pow(q / t, 2);
            } else {
              sumFitness += Math.pow(q, 2);
            }
          }
        });
        initialFitness = sumFitness;
        finalFitness = sumFitness;
        fitnessHistory = [sumFitness];
      }

      const fitnessImprovement = initialFitness > 0 ? ((initialFitness - finalFitness) / initialFitness) * 100 : 0;

      const baseName = scenarioName.trim() || "Unnamed Scenario";
      const existingCount = scenarios.filter(s => s.scenarioName.startsWith(baseName)).length;
      const uniqueName = existingCount > 0 ? `${baseName} (#${existingCount + 1})` : baseName;

      const newScenario: ScenarioResult = {
        scenarioName: uniqueName,
        initialFitness,
        finalFitness,
        fitnessImprovement,
        congestedEdges: congestedCount,
        fitnessHistory,
      };

      setScenarios(prev => [...prev, newScenario]);
    } catch (err: any) {
      setError(err.message || 'Optimization failed.');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !network) {
    return <div className="flex justify-center items-center h-screen">Loading network data...</div>;
  }

  if (error) {
    return <div className="p-4 bg-red-100 text-red-700 rounded m-4">{error}</div>;
  }

  if (!network) return null;

  // Compute real-time thresholds for visual components
  const currentThresholds: Record<string, number> = {};
  network.edges.forEach(edge => {
    currentThresholds[edge.id] = calculateThreshold(
      capacities[edge.id] ?? edge.capacity,
      edge.speed,
      edge.length,
      edge.road_type,
      edge.is_reference,
      networkType
    );
  });

  return (
    <div className="container mx-auto p-4 bg-gray-50 min-h-screen">
      <header className="flex justify-between items-center py-6 mb-4 border-b border-gray-200">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            {networkType === 'aditya' ? "Aditya's Network" : "Vedant's Network"} Traffic Optimization
          </h1>
          <p className="text-gray-500 mt-1">Research Prototype ({networkType === 'aditya' ? '4 Nodes' : '5 Nodes'})</p>
        </div>
        <button
          onClick={onBack}
          className="px-4 py-2 bg-gray-800 text-gray-100 border border-gray-700 rounded-lg shadow hover:bg-gray-700 hover:text-white font-medium cursor-pointer transition flex items-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path>
          </svg>
          Back to Networks
        </button>
      </header>
      
      <CapacityConfig 
        edges={network.edges} 
        capacities={capacities} 
        predictions={predictions}
        onCapacityChange={handleCapacityChange} 
        onPredictionChange={handlePredictionChange}
      />

      <div className="flex justify-between items-center mb-6 bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
        <div className="text-gray-700 font-medium">
          Optimization target: <span className="font-bold text-gray-900 bg-blue-50 text-blue-800 px-3 py-1.5 rounded-full text-sm">"{scenarioName}"</span>
        </div>

        <button
          onClick={handleOptimize}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded shadow disabled:opacity-50 flex items-center text-lg cursor-pointer"
        >
          {loading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Running...
            </>
          ) : (
            'Run Optimization'
          )}
        </button>
      </div>

      {/* Optimization Summary - shown only if optimization was run */}
      {result && (
        <OptimizationSummary result={result} />
      )}

      {/* Graphs Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <h2 className="text-xl font-bold mb-4 text-center text-gray-800">Graph Before Optimization</h2>
          <NetworkGraph 
            edgesData={network.edges} 
            congestionData={predictions} 
            thresholds={currentThresholds}
            selectedEdgeId={selectedEdgeId}
            onEdgeClick={(edgeId) => setSelectedEdgeId(edgeId || null)}
            onPredictionChange={handlePredictionChange}
          />
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <h2 className="text-xl font-bold mb-4 text-center text-gray-800">Graph After Optimization</h2>
          {result && result.optimization_triggered ? (
            <NetworkGraph 
              edgesData={result.graph_after.edges} 
              congestionData={result.optimized_congestion} 
              thresholds={result.thresholds}
              selectedEdgeId={selectedEdgeId}
              onEdgeClick={(edgeId) => setSelectedEdgeId(edgeId || null)}
              onPredictionChange={handlePredictionChange}
            />
          ) : result && !result.optimization_triggered ? (
             <div className="flex justify-center items-center h-[400px] text-gray-500 font-medium">
               Network Operating Normally. Optimization skipped.
             </div>
          ) : (
            <div className="flex justify-center items-center h-[400px] text-gray-500 italic">
               Run optimization to see results
             </div>
          )}
        </div>
      </div>

      {/* Congestion Table */}
      <div className="bg-white p-6 rounded-lg shadow border border-gray-200">
        <h2 className="text-xl font-bold mb-2 text-gray-800">Traffic Congestion Table</h2>
        <CongestionTable 
          edges={network.edges}
          capacities={capacities}
          predicted={predictions} 
          thresholds={currentThresholds}
          optimized={result?.optimized_congestion}
          greenTimes={result?.green_times}
        />
      </div>

      {/* Research Evaluation Metrics Table */}
      <ResearchMetricsTable 
        edges={network.edges}
        predictions={predictions}
        thresholds={currentThresholds}
        result={result}
      />

      {/* Research Analytics Section */}
      <ResearchAnalytics
        scenarios={scenarios}
        scenarioName={scenarioName}
        onChangeScenarioName={setScenarioName}
        onClearHistory={handleClearHistory}
      />
    </div>
  );
};
