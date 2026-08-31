import React, { useState, useEffect } from 'react';
import { api } from '../services/api';
import { NetworkResponse, OptimizationResponse } from '../types';
import { NetworkGraph } from './NetworkGraph';
import { calculateThreshold } from '../utils/threshold';

interface QaoaPsoComparisonProps {
  onBack: () => void;
}

export const QaoaPsoComparison: React.FC<QaoaPsoComparisonProps> = ({ onBack }) => {
  const [network, setNetwork] = useState<NetworkResponse | null>(null);
  const [psoResult, setPsoResult] = useState<OptimizationResponse | null>(null);
  const [qaoaResult, setQaoaResult] = useState<OptimizationResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Initial Predicted Loads (Aditya's 6 links - forward & reverse)
  const [predictions, setPredictions] = useState<Record<string, number>>({
    'A\u2192B': 1300, 'B\u2192A': 1300,
    'A\u2192C': 720,  'C\u2192A': 720,
    'A\u2192D': 595,  'D\u2192A': 595,
    'B\u2192C': 1360, 'C\u2192B': 1360,
    'B\u2192D': 510,  'D\u2192B': 510,
    'C\u2192D': 630,  'D\u2192C': 630,
  });

  const capacities: Record<string, number> = {
    'A\u2192B': 2000, 'B\u2192A': 2000,
    'A\u2192C': 1800, 'C\u2192A': 1800,
    'A\u2192D': 1700, 'D\u2192A': 1700,
    'B\u2192C': 2000, 'C\u2192B': 2000,
    'B\u2192D': 1700, 'D\u2192B': 1700,
    'C\u2192D': 1800, 'D\u2192C': 1800,
  };

  const canonicalEdges = [
    { key: 'A\u2192B', revKey: 'B\u2192A', label: 'Link A - B', capacity: 2000 },
    { key: 'A\u2192C', revKey: 'C\u2192A', label: 'Link A - C', capacity: 1800 },
    { key: 'A\u2192D', revKey: 'D\u2192A', label: 'Link A - D', capacity: 1700 },
    { key: 'B\u2192C', revKey: 'C\u2192B', label: 'Link B - C', capacity: 2000 },
    { key: 'B\u2192D', revKey: 'D\u2192B', label: 'Link B - D', capacity: 1700 },
    { key: 'C\u2192D', revKey: 'D\u2192C', label: 'Link C - D', capacity: 1800 },
  ];

  useEffect(() => {
    const initializeComparison = async () => {
      try {
        setLoading(true);
        setError(null);
        const netData = await api.getNetwork('aditya');
        setNetwork(netData);

        const [psoRes, qaoaRes] = await Promise.all([
          api.runOptimization({
            capacities,
            predictions,
            network_type: 'aditya',
          }),
          api.runQaoaOptimization({
            capacities,
            predictions,
          }),
        ]);

        setPsoResult(psoRes);
        setQaoaResult(qaoaRes);
      } catch (err: any) {
        setError(err.message || 'Failed to initialize comparison laboratory.');
      } finally {
        setLoading(false);
      }
    };

    initializeComparison();
  }, []);

  const handleRunOptimization = async () => {
    if (loading) return;
    try {
      setLoading(true);
      setError(null);

      const [psoRes, qaoaRes] = await Promise.all([
        api.runOptimization({
          capacities,
          predictions,
          network_type: 'aditya',
        }),
        api.runQaoaOptimization({
          capacities,
          predictions,
        }),
      ]);

      setPsoResult(psoRes);
      setQaoaResult(qaoaRes);
    } catch (err: any) {
      setError(err.message || 'Failed to execute parallel optimization comparison.');
    } finally {
      setLoading(false);
    }
  };

  const handlePredictionChange = (fwdKey: string, revKey: string, valStr: string) => {
    const val = parseFloat(valStr) || 0;
    setPredictions((prev) => ({
      ...prev,
      [fwdKey]: val,
      [revKey]: val,
    }));
  };

  const handleResetLoads = () => {
    const defaultPreds = {
      'A\u2192B': 1300, 'B\u2192A': 1300,
      'A\u2192C': 720,  'C\u2192A': 720,
      'A\u2192D': 595,  'D\u2192A': 595,
      'B\u2192C': 1360, 'C\u2192B': 1360,
      'B\u2192D': 510,  'D\u2192B': 510,
      'C\u2192D': 630,  'D\u2192C': 630,
    };
    setPredictions(defaultPreds);
  };

  // Thresholds calculation (60% of capacity)
  const currentThresholds: Record<string, number> = {};
  if (network) {
    network.edges.forEach(edge => {
      currentThresholds[edge.id] = calculateThreshold(
        capacities[edge.id] ?? edge.capacity,
        edge.speed,
        edge.length,
        edge.road_type,
        edge.is_reference,
        'aditya'
      );
    });
  }

  // Latency & Benchmark Metrics
  const psoLatency = psoResult?.latency_ms ?? 0;
  const qaoaLatency = qaoaResult?.latency_ms ?? 0;

  const isQaoaFaster = qaoaLatency > 0 && (psoLatency <= 0 || qaoaLatency < psoLatency);
  const latencyRatio = isQaoaFaster
    ? (psoLatency / Math.max(0.1, qaoaLatency)).toFixed(1)
    : (qaoaLatency / Math.max(0.1, psoLatency)).toFixed(1);

  const latencyBadgeText = isQaoaFaster
    ? `QAOA ${latencyRatio}x faster`
    : `PSO ${latencyRatio}x faster`;

  const calculatePeakPct = (res: OptimizationResponse | null, isBefore = false) => {
    if (!res) return 0;
    const flows = isBefore ? (res.before || predictions) : (res.after || predictions);
    const maxRatio = Math.max(...canonicalEdges.map(e => (flows[e.key] || 0) / e.capacity));
    return maxRatio * 100;
  };

  const peakBefore = calculatePeakPct(psoResult, true);
  const qaoaPeakAfter = calculatePeakPct(qaoaResult, false);
  const psoPeakAfter = calculatePeakPct(psoResult, false);

  const qaoaPeakReduction = peakBefore > 0 ? ((peakBefore - qaoaPeakAfter) / peakBefore) * 100 : 0;
  const psoPeakReduction = peakBefore > 0 ? ((peakBefore - psoPeakAfter) / peakBefore) * 100 : 0;

  const qaoaStdevBefore = qaoaResult?.initial_fitness ?? 0.1601;
  const qaoaStdevAfter = qaoaResult?.final_fitness ?? 0.0519;
  const qaoaLoadImprovement = qaoaResult?.load_distribution_improvement_pct ?? (
    qaoaStdevBefore > 0 ? ((qaoaStdevBefore - qaoaStdevAfter) / qaoaStdevBefore) * 100 : 0
  );

  const psoStdevBefore = psoResult?.initial_fitness ?? 0.1601;
  const psoStdevAfter = psoResult?.final_fitness ?? 0.0519;
  const psoLoadImprovement = psoStdevBefore > 0 ? ((psoStdevBefore - psoStdevAfter) / psoStdevBefore) * 100 : 0;

  if (loading && !network) {
    return <div className="flex justify-center items-center h-screen text-gray-600 font-medium">Loading comparison laboratory...</div>;
  }

  return (
    <div className="container mx-auto p-4 bg-gray-50 min-h-screen">
      {/* Header - Styled like Dashboard.tsx */}
      <header className="flex justify-between items-center py-6 mb-4 border-b border-gray-200">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">
            QAOA & PSO Traffic Optimization Comparison
          </h1>
          <p className="text-gray-500 mt-1">
            Parallel Execution Laboratory (Aditya's Network • 60% Capacity Threshold)
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleResetLoads}
            className="px-4 py-2 bg-white text-gray-700 border border-gray-300 rounded-lg shadow-sm hover:bg-gray-100 font-medium cursor-pointer transition flex items-center gap-2 text-sm"
          >
            <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Reset Baseline Loads
          </button>

          <button
            onClick={onBack}
            className="px-4 py-2 bg-gray-800 text-gray-100 border border-gray-700 rounded-lg shadow hover:bg-gray-700 hover:text-white font-medium cursor-pointer transition flex items-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            Back to Networks
          </button>
        </div>
      </header>

      {error && (
        <div className="p-4 bg-red-100 text-red-700 rounded-lg mb-6 border border-red-200">
          {error}
        </div>
      )}

      {/* 1. Demand Load Configuration Card - Styled like CapacityConfig.tsx */}
      <div className="bg-white p-6 rounded-lg shadow border border-gray-200 mb-6">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-xl font-bold text-gray-800">Predicted Demand Traffic Configuration (veh/hr)</h2>
            <p className="text-sm text-gray-500 mt-0.5">
              Edit predicted link loads below, then click <strong>Run Comparison Optimization</strong> to re-evaluate both QAOA and PSO models.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto mb-4">
          <table className="w-full text-sm text-left text-gray-500">
            <thead className="text-xs text-gray-700 uppercase bg-gray-50">
              <tr>
                <th scope="col" className="px-4 py-3">Edge Link</th>
                <th scope="col" className="px-4 py-3">Design Capacity (veh/hr)</th>
                <th scope="col" className="px-4 py-3">Operating Threshold (60%)</th>
                <th scope="col" className="px-4 py-3">Predicted Traffic Flow (veh/hr)</th>
                <th scope="col" className="px-4 py-3 text-right">Initial Occupancy (%)</th>
              </tr>
            </thead>
            <tbody>
              {canonicalEdges.map((edge) => {
                const currentVal = predictions[edge.key] || 0;
                const thresh = 0.60 * edge.capacity;
                const occPct = ((currentVal / edge.capacity) * 100).toFixed(1);
                const isOver = currentVal > thresh;

                return (
                  <tr key={edge.key} className="bg-white border-b hover:bg-gray-50">
                    <td className="px-4 py-3 font-medium text-gray-900">{edge.label} ({edge.key})</td>
                    <td className="px-4 py-3 font-semibold text-gray-700">{edge.capacity} v/h</td>
                    <td className="px-4 py-3 text-gray-600 font-mono">{thresh} v/h</td>
                    <td className="px-4 py-3">
                      <input
                        type="number"
                        min="0"
                        step="50"
                        className="border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500 w-44 text-gray-800 font-semibold"
                        value={currentVal}
                        onChange={(e) => handlePredictionChange(edge.key, edge.revKey, e.target.value)}
                      />
                    </td>
                    <td className="px-4 py-3 text-right font-mono font-bold">
                      <span className={`px-2.5 py-1 rounded text-xs ${isOver ? 'bg-amber-100 text-amber-800 border border-amber-300' : 'bg-gray-100 text-gray-700'}`}>
                        {occPct}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. Run Optimization Action Bar - Styled like Dashboard.tsx */}
      <div className="flex justify-between items-center mb-6 bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
        <div className="text-gray-700 font-medium">
          Optimization target: <span className="font-bold text-purple-900 bg-purple-50 text-purple-800 px-3 py-1.5 rounded-full text-sm">"QAOA Quantum vs. PSO Swarm Parallel Benchmark"</span>
        </div>

        <button
          onClick={handleRunOptimization}
          disabled={loading}
          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded shadow disabled:opacity-50 flex items-center text-lg cursor-pointer transition"
        >
          {loading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Running Parallel Optimization...
            </>
          ) : (
            'Run Comparison Optimization'
          )}
        </button>
      </div>

      {/* 3. Key Performance Benchmark Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {/* Card 1: Execution Latency */}
        <div className="bg-white p-6 rounded-lg shadow border border-gray-200">
          <div className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">
            Execution Latency Benchmark (ms)
          </div>
          <div className="flex items-baseline justify-between mt-2">
            <div>
              <div className="text-2xl font-black text-purple-700 font-mono">
                {qaoaLatency.toFixed(2)} ms
              </div>
              <div className="text-xs text-gray-500 font-medium mt-0.5">QAOA Rerouting</div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-black text-blue-700 font-mono">
                {psoLatency.toFixed(2)} ms
              </div>
              <div className="text-xs text-gray-500 font-medium mt-0.5">PSO Swarm Engine</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-gray-100 text-xs flex justify-between text-gray-600">
            <span>Performance Analysis</span>
            <span className="text-emerald-700 font-bold font-mono">
              {latencyBadgeText}
            </span>
          </div>
        </div>

        {/* Card 2: Peak Reduction (%) */}
        <div className="bg-white p-6 rounded-lg shadow border border-gray-200">
          <div className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">
            Peak Bottleneck Reduction (%)
          </div>
          <div className="flex items-baseline justify-between mt-2">
            <div>
              <div className="text-2xl font-black text-purple-700 font-mono">
                {qaoaPeakReduction.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-500 font-medium mt-0.5">
                QAOA ({peakBefore.toFixed(1)}% → {qaoaPeakAfter.toFixed(1)}%)
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-black text-blue-700 font-mono">
                {psoPeakReduction.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-500 font-medium mt-0.5">
                PSO ({peakBefore.toFixed(1)}% → {psoPeakAfter.toFixed(1)}%)
              </div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-gray-100 text-xs flex justify-between text-gray-600">
            <span>Bottleneck Difference</span>
            <span className="text-blue-700 font-bold font-mono">
              {Math.abs(qaoaPeakReduction - psoPeakReduction).toFixed(2)}% variance
            </span>
          </div>
        </div>

        {/* Card 3: Load Distribution Improvement */}
        <div className="bg-white p-6 rounded-lg shadow border border-gray-200">
          <div className="text-xs font-bold uppercase tracking-wider text-gray-500 mb-2">
            Load Distribution Improvement (%)
          </div>
          <div className="flex items-baseline justify-between mt-2">
            <div>
              <div className="text-2xl font-black text-purple-700 font-mono">
                {qaoaLoadImprovement.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-500 font-medium mt-0.5">
                QAOA (σ: {qaoaStdevBefore.toFixed(4)} → {qaoaStdevAfter.toFixed(4)})
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-black text-blue-700 font-mono">
                {psoLoadImprovement.toFixed(2)}%
              </div>
              <div className="text-xs text-gray-500 font-medium mt-0.5">
                PSO (σ: {psoStdevBefore.toFixed(4)} → {psoStdevAfter.toFixed(4)})
              </div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-gray-100 text-xs flex justify-between text-gray-600">
            <span>Variance Mitigation</span>
            <span className="text-emerald-700 font-bold font-mono">
              {qaoaLoadImprovement >= psoLoadImprovement ? 'QAOA Superior Balance' : 'PSO Superior Balance'}
            </span>
          </div>
        </div>
      </div>

      {/* 4. Dual Side-by-Side Network Graphs */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Left: QAOA Graph */}
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="flex justify-between items-center mb-3">
            <h2 className="text-xl font-bold text-gray-800">QAOA Optimization Graph</h2>
            <span className="bg-purple-100 text-purple-800 text-xs font-bold px-3 py-1 rounded-full">
              QAOA Rerouting State
            </span>
          </div>
          {network ? (
            <NetworkGraph
              edgesData={network.edges}
              congestionData={qaoaResult?.after || predictions}
              thresholds={currentThresholds}
            />
          ) : (
            <div className="flex justify-center items-center h-[400px] text-gray-500">
              Loading QAOA Graph...
            </div>
          )}
        </div>

        {/* Right: PSO Graph */}
        <div className="bg-white p-4 rounded-lg shadow border border-gray-200">
          <div className="flex justify-between items-center mb-3">
            <h2 className="text-xl font-bold text-gray-800">PSO Swarm Optimization Graph</h2>
            <span className="bg-blue-100 text-blue-800 text-xs font-bold px-3 py-1 rounded-full">
              PSO Timing State
            </span>
          </div>
          {network ? (
            <NetworkGraph
              edgesData={network.edges}
              congestionData={psoResult?.after || predictions}
              thresholds={currentThresholds}
            />
          ) : (
            <div className="flex justify-center items-center h-[400px] text-gray-500">
              Loading PSO Graph...
            </div>
          )}
        </div>
      </div>

      {/* 5. Comparative Results Table - Styled like ResearchMetricsTable.tsx */}
      <div className="bg-white p-6 rounded-lg shadow border border-gray-200 mb-8">
        <h2 className="text-xl font-bold mb-4 text-gray-800">System Optimization Performance Comparison Table</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left text-gray-500">
            <thead className="text-xs text-gray-700 uppercase bg-gray-50">
              <tr>
                <th scope="col" className="px-4 py-3">Performance Metric</th>
                <th scope="col" className="px-4 py-3">Initial Baseline</th>
                <th scope="col" className="px-4 py-3 text-purple-700">QAOA Method</th>
                <th scope="col" className="px-4 py-3 text-blue-700">PSO Swarm Method</th>
                <th scope="col" className="px-4 py-3 text-right">Comparison Analysis</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 font-mono">
              <tr className="bg-white hover:bg-gray-50">
                <td className="px-4 py-3 font-sans font-medium text-gray-900">Execution Latency</td>
                <td className="px-4 py-3 text-gray-500">0.00 ms</td>
                <td className="px-4 py-3 text-purple-700 font-bold">{qaoaLatency.toFixed(2)} ms</td>
                <td className="px-4 py-3 text-blue-700 font-bold">{psoLatency.toFixed(2)} ms</td>
                <td className="px-4 py-3 text-right font-sans text-emerald-700 font-bold">
                  {latencyBadgeText}
                </td>
              </tr>

              <tr className="bg-white hover:bg-gray-50">
                <td className="px-4 py-3 font-sans font-medium text-gray-900">Peak Congestion Bottleneck</td>
                <td className="px-4 py-3 text-amber-700 font-bold">{peakBefore.toFixed(1)}%</td>
                <td className="px-4 py-3 text-purple-700">{qaoaPeakAfter.toFixed(1)}%</td>
                <td className="px-4 py-3 text-blue-700">{psoPeakAfter.toFixed(1)}%</td>
                <td className="px-4 py-3 text-right font-sans text-gray-700">
                  {qaoaPeakAfter < psoPeakAfter
                    ? `QAOA Lower Peak (${qaoaPeakAfter.toFixed(1)}%)`
                    : psoPeakAfter < qaoaPeakAfter
                    ? `PSO Lower Peak (${psoPeakAfter.toFixed(1)}%)`
                    : `Equal Peak (${qaoaPeakAfter.toFixed(1)}%)`}
                </td>
              </tr>

              <tr className="bg-white hover:bg-gray-50">
                <td className="px-4 py-3 font-sans font-medium text-gray-900">Relative Peak Reduction</td>
                <td className="px-4 py-3 text-gray-500">0.00%</td>
                <td className="px-4 py-3 text-purple-700 font-bold">{qaoaPeakReduction.toFixed(2)}%</td>
                <td className="px-4 py-3 text-blue-700 font-bold">{psoPeakReduction.toFixed(2)}%</td>
                <td className="px-4 py-3 text-right font-sans font-purple-700 font-bold">
                  {qaoaPeakReduction > psoPeakReduction
                    ? `QAOA Superior (+${(qaoaPeakReduction - psoPeakReduction).toFixed(2)}%)`
                    : psoPeakReduction > qaoaPeakReduction
                    ? `PSO Superior (+${(psoPeakReduction - qaoaPeakReduction).toFixed(2)}%)`
                    : 'Equal Peak Reduction'}
                </td>
              </tr>

              <tr className="bg-white hover:bg-gray-50">
                <td className="px-4 py-3 font-sans font-medium text-gray-900">Network Load Variance (σ)</td>
                <td className="px-4 py-3 text-amber-700 font-bold">{qaoaStdevBefore.toFixed(4)}</td>
                <td className="px-4 py-3 text-purple-700">{qaoaStdevAfter.toFixed(4)}</td>
                <td className="px-4 py-3 text-blue-700">{psoStdevAfter.toFixed(4)}</td>
                <td className="px-4 py-3 text-right font-sans text-emerald-700 font-bold">
                  {qaoaStdevAfter < psoStdevAfter
                    ? `QAOA Tighter (σ: ${qaoaStdevAfter.toFixed(4)})`
                    : psoStdevAfter < qaoaStdevAfter
                    ? `PSO Tighter (σ: ${psoStdevAfter.toFixed(4)})`
                    : `Equal Variance (${qaoaStdevAfter.toFixed(4)})`}
                </td>
              </tr>

              <tr className="bg-white hover:bg-gray-50">
                <td className="px-4 py-3 font-sans font-medium text-gray-900">Overall Load Distribution Improvement</td>
                <td className="px-4 py-3 text-gray-500">0.00%</td>
                <td className="px-4 py-3 text-purple-700 font-bold">{qaoaLoadImprovement.toFixed(2)}%</td>
                <td className="px-4 py-3 text-blue-700 font-bold">{psoLoadImprovement.toFixed(2)}%</td>
                <td className="px-4 py-3 text-right font-sans text-emerald-700 font-bold">
                  {qaoaLoadImprovement > psoLoadImprovement
                    ? `QAOA Superior (+${(qaoaLoadImprovement - psoLoadImprovement).toFixed(2)}%)`
                    : psoLoadImprovement > qaoaLoadImprovement
                    ? `PSO Superior (+${(psoLoadImprovement - qaoaLoadImprovement).toFixed(2)}%)`
                    : `Equal Improvement (${qaoaLoadImprovement.toFixed(2)}%)`}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 6. Approach Green Times Comparison Table - Styled like CongestionTable.tsx */}
      <div className="bg-white p-6 rounded-lg shadow border border-gray-200 mb-8">
        <h2 className="text-xl font-bold mb-4 text-gray-800">Approach Green Times Comparison Table (per Intersection)</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {['A', 'B', 'C', 'D'].map((nodeId) => {
            const allKeys = ["A\u2192B", "A\u2192C", "A\u2192D", "B\u2192A", "B\u2192C", "B\u2192D", "C\u2192A", "C\u2192B", "C\u2192D", "D\u2192A", "D\u2192B", "D\u2192C"];
            const appKeys = allKeys.filter(k => k.startsWith(nodeId + '\u2192'));

            return (
              <div key={nodeId} className="border border-gray-200 rounded-lg p-4 bg-gray-50">
                <div className="flex justify-between items-center mb-3 pb-2 border-b border-gray-200">
                  <h3 className="font-bold text-gray-900 text-base">Node {nodeId} Intersection</h3>
                  <span className="text-xs text-gray-500 font-mono">Cycle: 90.0s</span>
                </div>

                <table className="w-full text-sm text-left text-gray-600">
                  <thead className="text-xs text-gray-700 uppercase bg-white">
                    <tr>
                      <th className="px-3 py-2">Approach</th>
                      <th className="px-3 py-2">Before Green</th>
                      <th className="px-3 py-2 text-purple-700">QAOA Green</th>
                      <th className="px-3 py-2 text-blue-700">PSO Green</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200 font-mono bg-white">
                    {appKeys.map((edgeKey) => {
                      const bg = psoResult?.green_times?.[edgeKey]?.old ?? 0.0;
                      const qaoaG = qaoaResult?.green_times?.[edgeKey]?.new ?? 0.0;
                      const psoG = psoResult?.green_times?.[edgeKey]?.new ?? 0.0;

                      return (
                        <tr key={edgeKey} className="hover:bg-gray-50">
                          <td className="px-3 py-2 font-sans font-medium text-gray-900">{edgeKey}</td>
                          <td className="px-3 py-2 text-gray-500">{bg.toFixed(1)}s</td>
                          <td className="px-3 py-2 text-purple-700 font-bold">{qaoaG.toFixed(1)}s</td>
                          <td className="px-3 py-2 text-blue-700 font-bold">{psoG.toFixed(1)}s</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
